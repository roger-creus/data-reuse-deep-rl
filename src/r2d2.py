import random
import time
import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from collections import deque

from utils.buffers import RolloutBuffer, SingleRolloutBuffer, PrioritizedRolloutBuffer
from utils.args import Args
from utils.models import QNetwork
from utils.utils import RecordEpisodeStatistics, linear_schedule
from IPython import embed

if __name__ == "__main__":
    import os
    import wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    
    # logging
    run_name = f"{args.env_id}_{args.exp_name}_usePER:{args.use_per}_s{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # environment
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        stack_num=1
    )
    envs.num_envs = args.num_envs
    envs = RecordEpisodeStatistics(envs)
    num_actions = envs.action_space.n
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # networks
    q_network = QNetwork(envs, lstm_hidden_size=args.lstm_hidden_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1e-3)
    
    target_network = QNetwork(envs, lstm_hidden_size=args.lstm_hidden_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    def update(data):
        B, T, C, H, W = data["observations"].shape
        x = data["observations"] / 255.0
        rtrn_extras = {"indices" : data["indices"]}
        
        def forward_pass(network, x, t, lstm_state, from_pixels=False):
            features = network.cnn(x[:, t].reshape(-1, C, H, W))
            features = network.mlp(features).reshape(B, -1)
            new_hidden, lstm_state = network.get_states(
                features, lstm_state, data["dones"], from_pixels=from_pixels
            )
            return new_hidden, lstm_state
            
        #### unroll online q-network and target q-network
        online_q_values = []
        target_q_values = []
        for t in range(T):
            if t == 0:
                lstm_state = (
                    data["lstm_states_cell"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                    data["lstm_states_hidden"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                )
                target_lstm_state = (
                    data["lstm_states_cell"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                    data["lstm_states_hidden"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                )
                
            # target net never uses gradients
            with torch.no_grad():
                target_hidden, target_lstm_state = forward_pass(target_network, x, t, target_lstm_state, from_pixels=False)
                
            # if during burn-in, online network doesnt use gradients
            if t < args.burn_in_lstm_length:
                with torch.no_grad():
                    new_hidden, lstm_state = forward_pass(q_network, x, t, lstm_state, from_pixels=False)
                    continue
            else:
                new_hidden, lstm_state = forward_pass(q_network, x, t, lstm_state, from_pixels=False)
            
            # compute q-values with online and target networks
            _online_q_values = q_network.q_func(new_hidden)
            
            # double q-learning
            with torch.no_grad():
                _online_q_values_max_actions = _online_q_values.argmax(dim=1)
                _target_q_values = target_network.q_func(target_hidden)
                _target_q_values_max = _target_q_values.gather(1, _online_q_values_max_actions.unsqueeze(-1)).squeeze(-1)
            
            # not double q-learning
            #_target_q_values_max = target_network.q_func(target_hidden).max(dim=1).values
            
            # store q-values
            online_q_values.append(_online_q_values)
            target_q_values.append(_target_q_values_max)
            
        # at this point we only have data for the last T - burn_in_lstm_length steps
        actions = data["actions"][:, args.burn_in_lstm_length:].long().unsqueeze(-1)
        rewards = data["rewards"][:, args.burn_in_lstm_length:]
        dones = data["dones"][:, args.burn_in_lstm_length:]
                
        online_q_values = torch.stack(online_q_values, dim=1) # B, T, num_actions
        target_q_values = torch.stack(target_q_values, dim=1) # B, T, num_actions
        
        # get q-values for actions taken
        online_q_values = online_q_values.gather(2, actions).squeeze(-1)
        
        # compute q_lambda returns
        with torch.no_grad():
            updte_steps = T - args.burn_in_lstm_length
            returns = torch.zeros(B, updte_steps, device=device)
            for t in reversed(range(updte_steps)):
                if t == updte_steps - 1:
                    # last step -- bootstrap q-values from out-of-bounds observation and done of the rollout (was also stored in the buffer!)
                    # this target_lstm_state could be also the last one from the sampled rollout (from the buffer) instead of the last one of the unrolled rollout 
                    target_hidden, _ = target_network.get_states(
                        data["last_observation"], target_lstm_state, data["last_done"]
                    )
                    target_q_values_last = target_network.q_func(target_hidden).max(dim=1).values
                    next_non_terminal = 1.0 - data["last_done"]
                    returns[:, t] = rewards[:, t] + args.gamma * target_q_values_last * next_non_terminal
                else:
                    next_non_terminal = 1.0 - dones[:, t+1]
                    returns[:, t] = rewards[:, t] + args.gamma * (
                        args.q_lambda * returns[:, t+1] + (1 - args.q_lambda) * target_q_values[:, t+1] * next_non_terminal
                    ) 

        # compute loss
        loss = (online_q_values - returns).pow(2).sum(dim=1)
        if args.use_per:
            loss = loss * data["importance_weights"]
        loss = loss.mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rtrn_extras["q_values"] = online_q_values.detach().mean()
        
        if args.use_per:
            # compute new priorities
            max_absolute_td_error = (online_q_values.detach() - returns).abs().max(dim=1).values
            mean_absolute_td_error = (online_q_values.detach() - returns).abs().mean(dim=1)
            priorities = args.per_eta * max_absolute_td_error + (1 - args.per_eta) * mean_absolute_td_error
            priorities = priorities.cpu().numpy()
            rtrn_extras["priorities"] = priorities
        
        return loss.detach(), rtrn_extras

    # replay buffer
    rb_clss = PrioritizedRolloutBuffer if args.use_per else RolloutBuffer
    rb_args = dict(alpha=args.per_alpha,beta=args.per_beta) if args.use_per else {}
    
    replay_buffer = rb_clss(
        num_rollouts=args.num_rollouts,
        rollout_length=args.rollout_length,
        lstm_hidden_size=args.lstm_hidden_size,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=args.num_envs,
        **rb_args
    )
    
    single_rollout_buffer = SingleRolloutBuffer(
        rollout_length=args.rollout_length,
        lstm_hidden_size=args.lstm_hidden_size,
        n_envs=args.num_envs,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
    )
    
    # start
    start_time = time.time()
    global_step = 0
    train_step = 0
    avg_returns = deque(maxlen=20)
    obs, _ = envs.reset()
    done = np.zeros(args.num_envs, dtype=np.uint8)
    
    # initial lstm states
    lstm_state = (
        torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size).to(device),
        torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size).to(device),
    )

    while global_step < args.total_timesteps:
        # increment global step
        global_step += envs.num_envs
        
        # anneal beta for prioritized replay
        per_beta = linear_schedule(args.per_beta, 1.0, args.total_timesteps, global_step)
        replay_buffer.beta = per_beta
        
        # select actions
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        epsilon_tensor = torch.as_tensor(epsilon, device=device, dtype=torch.float)
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float)
        done_tensor = torch.as_tensor(done, device=device, dtype=torch.float)

        with torch.no_grad():
            q_values, next_lstm_state = q_network(obs_tensor, lstm_state, done_tensor)
            q_actions = q_values.argmax(dim=1)
            random_actions = torch.randint(0, envs.action_space.n, (args.num_envs,)).to(device)
            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            actions = torch.where(explore, random_actions, q_actions)
        
        actions = actions.cpu().numpy() 

        # execute actions
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # log data
        _avg_ep_returns = []
        for idx, d in enumerate(np.logical_or(terminations, truncations)):
            if d and infos["lives"][idx] == 0:
                avg_returns.append(infos["r"][idx])
                _avg_ep_returns.append(infos["r"][idx])
                
        if len(_avg_ep_returns) > 0:
            print(f"global_step={global_step}, avg_ep_return={np.mean(_avg_ep_returns)}")

        # add data to single rollout buffer
        single_rollout_buffer.add(
            observations=obs.astype(np.uint8),
            actions=actions.astype(np.uint8),
            rewards=rewards,
            dones=done.astype(np.uint8),
            lstm_states=lstm_state,
        )

        # crucial step easy to overlook
        obs = next_obs
        lstm_state = next_lstm_state
        done = np.logical_or(terminations, truncations)
        
        # add complete rollouts to replay buffer
        if single_rollout_buffer.full:
            replay_buffer.add(
                observations=single_rollout_buffer.observations,
                actions=single_rollout_buffer.actions,
                rewards=single_rollout_buffer.rewards,
                dones=single_rollout_buffer.dones,
                lstm_states=single_rollout_buffer.lstm_states,
                last_observation=torch.from_numpy(obs),
                last_done=torch.from_numpy(done),
            )
            single_rollout_buffer.reset_half()

        # train the model
        if global_step > args.learning_starts:
            if (global_step / args.num_envs) % args.train_frequency == 0:
                # perform 1 training step and update priorities
                train_step += 1
                data = replay_buffer.sample(args.batch_size)
                loss, out = update(data)
                
                if args.use_per:
                    replay_buffer.update_priorities(out["indices"], out["priorities"])

                if train_step % args.log_every == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    if args.track:
                        wandb.log({
                            "train/loss": loss.cpu().item(),
                            "schedule/epsilon": epsilon,
                            "train/avg_ep_return": np.mean(avg_returns),
                            "train/q_values": out["q_values"].cpu().item(),
                            "global_step": global_step,
                            "train_step": train_step,
                            "sps": int(global_step / (time.time() - start_time)),
                        }, step=global_step)

            # update target network
            if train_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()