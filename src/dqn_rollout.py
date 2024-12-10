import random
import time

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import os
import wandb
from utils.buffers import RolloutBuffer, SingleRolloutBuffer
from collections import deque
from utils.args import Args
from utils.models import QNetwork
from utils.utils import RecordEpisodeStatistics, linear_schedule
from utils.hns import _compute_hns
from utils.metrics import compute_training_metrics
from IPython import embed

if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    # args
    args = tyro.cli(Args)
    total_train_steps = ((args.total_timesteps - args.learning_starts) // args.num_envs // args.train_frequency) * args.num_grad_steps
    steps_to_log = np.linspace(0, total_train_steps-1, args.num_logs, dtype=int)
    steps_to_log = np.insert(steps_to_log, 0, 1)
    steps_to_log = set(steps_to_log)
    rr = args.num_grad_steps / args.train_frequency
    drr = ((args.batch_size * args.rollout_length) / args.num_envs) * rr
    args.rr = rr
    args.drr = drr
    
    if args.exp_name == None:
        args.exp_name = "dqn_rollout"
    
    if args.use_n_steps:
        train_objective = "n_steps"
    elif args.use_q_lambda:
        train_objective = "q_lambda"
    else:
        raise ValueError(f"Please select a training objective: `use_n_steps` or `use_q_lambda`")
    
    # adjust buffer size
    args.buffer_size = args.buffer_size // args.rollout_length
        
    run_name = f"algo:{args.exp_name}_env:{args.env_id}_objective:{train_objective}_nenvs:{args.num_envs}_batchSize:{args.batch_size}_rolloutLength:{args.rollout_length}_numGradSteps:{args.num_grad_steps}_RR:{rr}_DRR:{drr}_seed:{args.seed}_{int(time.time())}"
    args.run_name = run_name
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )
    # TRY NOT TO MODIFY: seeding
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
        stack_num=args.input_channels,
    )
    envs.num_envs = args.num_envs
    envs = RecordEpisodeStatistics(envs)
    num_actions = envs.action_space.n
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # network
    q_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # for logging metrics
    old_q_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    old_q_network.load_state_dict(q_network.state_dict())

    # large rollout buffer
    replay_buffer = RolloutBuffer(
        buffer_size=args.buffer_size,
        rollout_length=args.rollout_length,
        lstm_hidden_size=args.lstm_hidden_size,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=args.num_envs,
    )
    
    single_rollout_buffer = SingleRolloutBuffer(
        rollout_length=args.rollout_length,
        lstm_hidden_size=args.lstm_hidden_size,
        n_envs=args.num_envs,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
    )

    # start tr
    start_time = time.time()
    obs, _ = envs.reset()
    global_step = 0
    train_step = 0
    avg_returns = deque(maxlen=50)
    done = np.zeros(args.num_envs, dtype=np.uint8)
    while global_step < args.total_timesteps:
        global_step += args.num_envs
        
        # select actions
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float)
            random_actions = torch.randint(0, envs.action_space.n, (args.num_envs,)).to(device)
            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            actions = torch.where(explore, random_actions, q_network(obs_tensor).argmax(dim=1))
            actions = actions.cpu().numpy()

        # execute the game and log data.
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
            lstm_states=None,
        )

        # CRUCIAL step easy to overlook
        obs = next_obs
        done = np.logical_or(terminations, truncations)
        
        # add complete rollouts to replay buffer
        if single_rollout_buffer.full:
            replay_buffer.add(
                observations=single_rollout_buffer.observations,
                actions=single_rollout_buffer.actions,
                rewards=single_rollout_buffer.rewards,
                dones=single_rollout_buffer.dones,
                last_observation=torch.from_numpy(obs),
                last_done=torch.from_numpy(done),
            )
            single_rollout_buffer.reset_half()

        # optimize the model
        if global_step > args.learning_starts:
            if (global_step / args.num_envs) % args.train_frequency == 0:
                for _ in range(args.num_grad_steps):
                    old_q_network.load_state_dict(q_network.state_dict())
                    
                    train_step += 1
                    data = replay_buffer.sample(args.batch_size)
                    B, T, C, H, W = data["observations"].shape
                    x = data["observations"]
                    
                    # compute value estimates
                    q_values = q_network(x.view(B*T, C, H, W)).view(B, T, -1) # B x T x A
                    q_values = q_values.gather(2, data["actions"].long().unsqueeze(-1)).squeeze(-1)
                    
                    # compute target values
                    with torch.no_grad():
                        target_q_values = target_network(data["observations"].view(B*T, C, H, W)).view(B, T, -1).max(dim=-1)[0]
                        
                    if args.use_q_lambda:
                        with torch.no_grad():
                            returns = torch.zeros(B, T).to(device)
                            for t in reversed(range(T)):
                                if t == T - 1:
                                    # last step -- bootstrap q-values from out-of-bounds observation and done of the rollout (was also stored in the buffer!)
                                    # this target_lstm_state could be also the last one from the sampled rollout (from the buffer) instead of the last one of the unrolled rollout 
                                    target_q_values_last = target_network(data["last_observation"]).max(dim=1).values
                                    next_non_terminal = 1.0 - data["last_done"]
                                    returns[:, t] = data["rewards"][:, t] + args.gamma * target_q_values_last * next_non_terminal
                                else:
                                    next_non_terminal = 1.0 - data["dones"][:, t+1]
                                    returns[:, t] = data["rewards"][:, t] + args.gamma * (
                                        args.q_lambda * returns[:, t+1] + (1 - args.q_lambda) * target_q_values[:, t+1] 
                                    ) * next_non_terminal
                                    
                    
                    elif args.use_n_steps:
                        n = 5  # n-step bootstrapping
                        with torch.no_grad():
                            # Precompute discounts for efficiency
                            discounts = args.gamma ** torch.arange(0, n, device=device)
                            
                            # Initialize returns
                            returns = torch.zeros(B, T).to(device)
                            
                            # Compute n-step returns
                            for t in reversed(range(T)):
                                if t == T - 1:
                                    # Last step: Use rewards and bootstrap if not done
                                    target_q_values_last = target_network(data["last_observation"]).max(dim=1).values
                                    next_non_terminal = 1.0 - data["last_done"]
                                    returns[:, t] = (
                                        data["rewards"][:, t]
                                        + args.gamma * target_q_values_last * next_non_terminal
                                    )
                                else:
                                    # Collect rewards within the n-step window
                                    n_steps = min(n, T - t)
                                    rewards = data["rewards"][:, t : t + n_steps]
                                    dones = data["dones"][:, t : t + n_steps]
                                    
                                    # Compute cumulative rewards
                                    valid_mask = torch.cumprod(1.0 - dones, dim=1)
                                    masked_rewards = rewards * valid_mask
                                    
                                    step_discounts = discounts[:n_steps]
                                    cumulative_rewards = torch.sum(masked_rewards * step_discounts, dim=1)
                                    
                                    # Add bootstrapped Q-values for the last step within n-steps
                                    bootstrap_idx = t + n
                                    if bootstrap_idx < T:
                                        cumulative_rewards += (
                                            discounts[n_steps - 1]
                                            * target_q_values[:, bootstrap_idx]
                                            * (1.0 - data["dones"][:, bootstrap_idx])
                                        )
                                    
                                    # Combine with returns of the next step (if valid)
                                    next_non_terminal = 1.0 - data["dones"][:, t + 1]
                                    returns[:, t] = (
                                        data["rewards"][:, t]
                                        + args.gamma * returns[:, t + 1] * next_non_terminal
                                    )
                    else:
                        raise ValueError(f"Please select a training objective: `use_n_steps` or `use_q_lambda`")
                    
                    loss = (q_values - returns).pow(2).sum(-1).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    if train_step in steps_to_log:
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        if args.track:
                            metrics = compute_training_metrics(
                                agent=q_network,
                                old_agent=old_q_network,
                                obs=replay_buffer.sample(50)["observations"],
                            )
                            for k, v in metrics.items():
                                metrics[k] = v.cpu().item() if torch.is_tensor(v) else v
                            
                            wandb.log({
                                "train/loss": loss.detach().cpu().item(),
                                "schedule/epsilon": epsilon,
                                "train/avg_ep_return": np.mean(avg_returns),
                                "train/hns": _compute_hns(args.env_id, np.mean(avg_returns)),
                                "train/q_values": q_values.mean().cpu().item(),
                                "global_step": global_step,
                                "train_step": train_step,
                                "sps": int(global_step / (time.time() - start_time)),
                                **metrics
                            }, step=global_step)

                    # update target network
                    if train_step % args.target_network_frequency == 0:
                        print("Updating target network")
                        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                            target_network_param.data.copy_(
                                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                            )

    envs.close()