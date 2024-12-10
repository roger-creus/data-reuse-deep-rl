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
from stable_baselines3.common.buffers import ReplayBuffer
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
    drr = (args.batch_size / args.num_envs) * rr
    args.rr = rr
    args.drr = drr
    
    if args.exp_name == None:
        args.exp_name = "dqn_td1"
    
    run_name = f"algo:{args.exp_name}_env:{args.env_id}_nenvs:{args.num_envs}_batchSize:{args.batch_size}_numGradSteps:{args.num_grad_steps}_RR:{rr}_DRR:{drr}_seed:{args.seed}_{int(time.time())}"
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

    q_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # for logging metrics
    old_q_network = QNetwork(envs, input_channels=args.input_channels, lstm_hidden_size=args.lstm_hidden_size).to(device)
    old_q_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        n_envs=args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # start tr
    start_time = time.time()
    obs, _ = envs.reset()
    global_step = 0
    train_step = 0
    avg_returns = deque(maxlen=50)
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

        # add data to the replay buffer
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # CRUCIAL step easy to overlook
        obs = next_obs

        # optimize the model
        if global_step > args.learning_starts:
            if (global_step / args.num_envs) % args.train_frequency == 0:
                for _ in range(args.num_grad_steps):
                    old_q_network.load_state_dict(q_network.state_dict())

                    train_step += 1
                        
                    data = rb.sample(args.batch_size)
                    
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                        
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if train_step in steps_to_log:
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        if args.track:
                            metrics = compute_training_metrics(
                                agent=q_network,
                                old_agent=old_q_network,
                                obs=rb.sample(4096).observations,
                            )
                            for k, v in metrics.items():
                                metrics[k] = v.cpu().item() if torch.is_tensor(v) else v
                            
                            wandb.log({
                                "train/loss": loss.detach().cpu().item(),
                                "schedule/epsilon": epsilon,
                                "train/avg_ep_return": np.mean(avg_returns),
                                "train/hns": _compute_hns(args.env_id, np.mean(avg_returns)),
                                "train/q_values": old_val.mean().cpu().item(),
                                "global_step": global_step,
                                "train_step": train_step,
                                "sps": int(global_step / (time.time() - start_time)),
                                **metrics
                            }, step=global_step)

                    # update target network
                    if train_step % args.target_network_frequency == 0:
                        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                            target_network_param.data.copy_(
                                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                            )

    envs.close()