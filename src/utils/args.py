from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "r2d2"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Experiment
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    
    # Data
    num_envs: int = 256
    """the number of parallel game environments"""
    num_rollouts: int = 5000
    """the replay memory buffer size"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    rollout_length: int = 80
    """the rollout length of the R2D2 algorithm"""
    
    # Exploration
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    
    # Training
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    learning_starts: int = 200000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training the networks"""
    q_lambda: float = 0.65
    """the lambda value of the Q(lambda) algorithm"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    
    # LSTM
    lstm_hidden_size: int = 512
    """the hidden size of the LSTM network"""
    burn_in_lstm_length: int = 40
    """the burn-in length of the LSTM network"""
    
    # Prioritized Experience Replay
    use_per: bool = True
    """if toggled, the prioritized experience replay will be used"""
    per_eta: float = 0.9
    """the priority eta value for updating the priorities"""
    per_alpha: float = 0.9
    """the priority alpha value for updating the priorities"""
    per_beta: float = 0.6
    """the priority beta value for updating the priorities"""
    
    # logging
    log_every: int = 25
    """the interval of logging in train steps"""