import random
import time
import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
import os
import wandb
from torch import multiprocessing as mp
from collections import deque

from utils.buffers import RolloutBuffer, SingleRolloutBuffer, PrioritizedRolloutBuffer
from utils.args import Args
from utils.models import QNetwork
from utils.utils import RecordEpisodeStatistics, linear_schedule
from IPython import embed

if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["OMP_NUM_THREADS"] = "1" 
    virtual_cpu_count = mp.cpu_count() - 1

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

    mock_envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=1,
        thread_affinity_offset=0,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        stack_num=1
    )

    # networks
    q_network = QNetwork(mock_envs, lstm_hidden_size=args.lstm_hidden_size).to(device).share_memory()
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1e-3)
    target_network = QNetwork(mock_envs, lstm_hidden_size=args.lstm_hidden_size).to(device).share_memory()
    target_network.load_state_dict(q_network.state_dict())
    
    # replay buffer
    rb_clss = PrioritizedRolloutBuffer if args.use_per else RolloutBuffer
    rb_args = dict(alpha=args.per_alpha,beta=args.per_beta) if args.use_per else {}
    
    replay_buffer = rb_clss(
        num_rollouts=args.num_rollouts,
        rollout_length=args.rollout_length,
        lstm_hidden_size=args.lstm_hidden_size,
        observation_space=mock_envs.observation_space,
        action_space=mock_envs.action_space,
        device=device,
        n_envs=args.num_envs,
        **rb_args
    )
    
    mock_envs.close()
    
    # Training function
    def update(learner_idx, replay_buffer):
        torch.set_num_threads(1)
        
        while True:
            data = replay_buffer.sample(args.batch_size)
            print(f"learner={learner_idx}, data={data}")
            if data is not None:
                B, T, C, H, W = data["observations"].shape
                x = data["observations"] / 255.0
                rtrn_extras = {"indices" : data["indices"]}
                
                # utility to unroll the online and target networks over a rollout
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
                    # init lstm state from the buffer
                    if t == 0:
                        lstm_state = (
                            data["lstm_states_cell"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                            data["lstm_states_hidden"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                        )
                        target_lstm_state = (
                            data["lstm_states_cell"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                            data["lstm_states_hidden"][:, t].unsqueeze(0), # 1, B, lstm_hidden_size
                        )
                        
                    # unroll target network without gradients
                    with torch.no_grad():
                        target_hidden, target_lstm_state = forward_pass(target_network, x, t, target_lstm_state, from_pixels=False)
                        
                    # unroll online network - during burn-in, online network doesnt use gradients
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
                    
                    # store q-values
                    online_q_values.append(_online_q_values)
                    target_q_values.append(_target_q_values_max)
                    
                # at this point we only have training data for the last T - burn_in_lstm_length steps
                actions = data["actions"][:, args.burn_in_lstm_length:].long().unsqueeze(-1)
                rewards = data["rewards"][:, args.burn_in_lstm_length:]
                dones = data["dones"][:, args.burn_in_lstm_length:]
                online_q_values = torch.stack(online_q_values, dim=1) # B, T, num_actions
                online_q_values = online_q_values.gather(2, actions).squeeze(-1)
                target_q_values = torch.stack(target_q_values, dim=1) # B, T, num_actions
                
                # compute q_lambda returns withoiut gradients
                with torch.no_grad():
                    updte_steps = T - args.burn_in_lstm_length
                    returns = torch.zeros(B, updte_steps, device=device)
                    for t in reversed(range(updte_steps)):
                        if t == updte_steps - 1:
                            # last step -- bootstrap q-values from out-of-bounds observation and done of the rollout (was also stored in the buffer!)
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
                # importance sampling weights if using PER
                if args.use_per:
                    loss = loss * data["importance_weights"]
                loss = loss.mean()
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rtrn_extras["q_values"] = online_q_values.detach().mean()
                
                # update priorities if using PER
                if args.use_per:
                    max_absolute_td_error = (online_q_values.detach() - returns).abs().max(dim=1).values
                    mean_absolute_td_error = (online_q_values.detach() - returns).abs().mean(dim=1)
                    priorities = args.per_eta * max_absolute_td_error + (1 - args.per_eta) * mean_absolute_td_error
                    priorities = priorities.cpu().numpy()
                    replay_buffer.update_priorities(data["indices"], priorities)
                    
                print(f"learner={learner_idx}, loss={loss.item()}")
                
                # update target network
                #if train_step % args.target_network_frequency == 0:
                #    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                #        target_network_param.data.copy_(
                #            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                #        )

    
    def actor_process(actor_idx, replay_buffer):
        torch.set_num_threads(virtual_cpu_count - args.num_learners)
        print(f"Started actor process with {virtual_cpu_count - args.num_learners} threads")
        
        # environment
        envs = envpool.make(
            args.env_id,
            env_type="gymnasium",
            num_envs=args.num_envs,
            batch_size=args.num_envs,
            num_threads=virtual_cpu_count - args.num_learners,
            thread_affinity_offset=args.num_learners,
            episodic_life=True,
            reward_clip=True,
            seed=args.seed,
            stack_num=1
        )
        envs.num_envs = args.num_envs
        envs = RecordEpisodeStatistics(envs)
        assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        
        
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
        avg_returns = deque(maxlen=20)
        obs, _ = envs.reset()
        done = np.zeros(args.num_envs, dtype=np.uint8)
        
        # initial lstm states
        lstm_state = (
            torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size).to(device),
            torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size).to(device),
        )
        
        
        # ACTORS LOOP
        while True:
            # increment global step
            global_step += envs.num_envs
            
            # select actions
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
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
            
    # ACTOR LOOP
    actor_thread = mp.Process(target=actor_process, args=(0, replay_buffer))
    actor_thread.start()

    # LEARNER LOOP
    learners = []
    for learner_idx in range(args.num_learners):
        p = mp.Process(target=update, args=(learner_idx, replay_buffer))
        learners.append(p)
        p.start()
    
    # Join processes
    actor_thread.join()
    for p in learners:
        p.join()