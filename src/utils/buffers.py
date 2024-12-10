import torch
import random
import numpy as np
import threading
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from torch import multiprocessing as mp
from IPython import embed
from utils.segment_tree import SumSegmentTree, MinSegmentTree

class PrioritizedRolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        rollout_length: int,
        lstm_hidden_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs: int = 1,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        async_: bool = False,
    ):
        self.buffer_size = buffer_size
        self.rollout_length = rollout_length
        self.lstm_hidden_size = lstm_hidden_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.n_envs = n_envs
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.async_ = async_

        # Initialize data buffers
        self.reset()

        # Initialize segment trees
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
        
        if self.async_:
            self.lock = mp.Lock()
            self.data_ready_event = mp.Event()
            
    def reset(self) -> None:
        self.observations = torch.zeros((self.buffer_size, self.rollout_length, *self.obs_shape), dtype=torch.uint8, device='cpu')
        self.actions = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.uint8, device='cpu')
        self.rewards = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.float32, device='cpu')
        self.dones = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.uint8, device='cpu')
        self.last_observation = torch.zeros((self.buffer_size, *self.obs_shape), dtype=torch.uint8, device='cpu')
        self.last_done = torch.zeros(self.buffer_size, dtype=torch.uint8, device='cpu')
        self.lstm_states = {
            "hidden": torch.zeros((self.buffer_size, self.rollout_length, self.lstm_hidden_size), dtype=torch.float32, device='cpu'),
            "cell": torch.zeros((self.buffer_size, self.rollout_length, self.lstm_hidden_size), dtype=torch.float32, device='cpu'),
        }
        
        # put in shared memory if async
        if self.async_:
            self.observations.share_memory_()
            self.actions.share_memory_()
            self.rewards.share_memory_()
            self.dones.share_memory_()
            self.last_observation.share_memory_()
            self.last_done.share_memory_()
            self.lstm_states["hidden"].share_memory_()
            self.lstm_states["cell"].share_memory_()
        
        self.pos = 0
        self.full = False

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        last_observation: torch.Tensor,
        last_done: torch.Tensor,
        lstm_states: Dict[str, torch.Tensor],
        priority: float = 1.0,
    ):
        
        if self.async_:
            self.lock.acquire()
            self._add(
                observations, actions, rewards, dones, last_observation, last_done, lstm_states, priority
            )
            self.lock.release()
            self.data_ready_event.set()
            
        else:
            self._add(
                observations, actions, rewards, dones, last_observation, last_done, lstm_states, priority
            )
            
    def _add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        last_observation: torch.Tensor,
        last_done: torch.Tensor,
        lstm_states: Dict[str, torch.Tensor],
        priority: float,
    ):
        start_idx = self.pos
        end_idx = (self.pos + self.n_envs) % self.buffer_size

        obs = observations.transpose(0, 1).to('cpu')
        act = actions.transpose(0, 1).to('cpu')
        rew = rewards.transpose(0, 1).to('cpu')
        don = dones.transpose(0, 1).to('cpu')
        lstm_sts_hidden = lstm_states["hidden"].transpose(0, 1).to('cpu')
        lstm_sts_cell = lstm_states["cell"].transpose(0, 1).to('cpu')

        # Handle circular buffer logic
        if end_idx > start_idx:
            self.observations[start_idx:end_idx] = obs
            self.actions[start_idx:end_idx] = act
            self.rewards[start_idx:end_idx] = rew
            self.dones[start_idx:end_idx] = don
            self.last_observation[start_idx:end_idx] = last_observation
            self.last_done[start_idx:end_idx] = last_done
            self.lstm_states["hidden"][start_idx:end_idx] = lstm_sts_hidden
            self.lstm_states["cell"][start_idx:end_idx] = lstm_sts_cell
        else:
            split_point = self.buffer_size - start_idx
            self.observations[start_idx:] = obs[:split_point]
            self.observations[:end_idx] = obs[split_point:]
            self.actions[start_idx:] = act[:split_point]
            self.actions[:end_idx] = act[split_point:]
            self.rewards[start_idx:] = rew[:split_point]
            self.rewards[:end_idx] = rew[split_point:]
            self.dones[start_idx:] = don[:split_point]
            self.dones[:end_idx] = don[split_point:]
            self.last_observation[start_idx:] = last_observation[:split_point]
            self.last_observation[:end_idx] = last_observation[split_point:]
            self.last_done[start_idx:] = last_done[:split_point]
            self.last_done[:end_idx] = last_done[split_point:]
            self.lstm_states["hidden"][start_idx:] = lstm_sts_hidden[:split_point]
            self.lstm_states["hidden"][:end_idx] = lstm_sts_hidden[split_point:]
            self.lstm_states["cell"][start_idx:] = lstm_sts_cell[:split_point]
            self.lstm_states["cell"][:end_idx] = lstm_sts_cell[split_point:]

        # Update priorities in segment trees
        for i in range(self.n_envs):
            idx = (start_idx + i) % self.buffer_size
            priority_val = priority ** self.alpha
            self.sum_tree[idx] = priority_val
            self.min_tree[idx] = priority_val
        
        self.max_priority = max(self.max_priority, priority)
        self.pos = end_idx
        self.full = self.full or end_idx < start_idx

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.async_:
            self.data_ready_event.wait()
            self.lock.acquire()
            data = self._sample(batch_size)
            self.lock.release()
            return data
        else:
            return self._sample(batch_size)
        
    def _sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        upper_bound = self.buffer_size if self.full else self.pos
        indices = self._sample_proportional(batch_size, upper_bound)

        # Calculate importance weights
        importance_weights = torch.tensor([
            self._calculate_weight(idx, self.beta, upper_bound) for idx in indices
        ], dtype=torch.float32, device=self.device)
        importance_weights /= importance_weights.max()

        return self._get_samples(indices, importance_weights)

    def _sample_proportional(self, batch_size: int, upper_bound: int):
        indices = []
        p_total = self.sum_tree.sum(0, upper_bound - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float, upper_bound: int):
        p_min = self.min_tree.min() / self.sum_tree.sum(0, upper_bound - 1)
        max_weight = (p_min * upper_bound) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum(0, upper_bound - 1)
        weight = (p_sample * upper_bound) ** (-beta)
        return weight / max_weight

    def _get_samples(self, indices: torch.Tensor, importance_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "importance_weights": importance_weights,
            "dones": self.dones[indices],
            "last_observation": self.last_observation[indices],
            "last_done": self.last_done[indices],
            "lstm_states_hidden": self.lstm_states["hidden"][indices],
            "lstm_states_cell": self.lstm_states["cell"][indices],
            "indices": indices,
        }
        data = {k: v.to(self.device) for k, v in data.items() if k != "indices"}
        data["indices"] = indices
        return data
        
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        if self.async_:
            self.lock.acquire()
            self._update_priorities(indices, priorities)
            self.lock.release()
        else:
            self._update_priorities(indices, priorities)
            
    def _update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        for idx, priority in zip(indices, priorities):
            priority_val = priority ** self.alpha
            self.sum_tree[idx] = priority_val
            self.min_tree[idx] = priority_val
            self.max_priority = max(self.max_priority, priority)
            
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable attributes
        if self.async_:
            del state['lock']
            del state['data_ready_event']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize unpickleable attributes
        if self.async_:
            self.lock = mp.Lock()
            self.data_ready_event = mp.Event()
            
class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        rollout_length: int,
        lstm_hidden_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.rollout_length = rollout_length
        self.lstm_hidden_size = lstm_hidden_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.n_envs = n_envs
        self.device = device

        # Initialize data buffers
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros((self.buffer_size, self.rollout_length, *self.obs_shape), dtype=torch.uint8, device='cpu')
        self.actions = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.uint8, device='cpu')
        self.rewards = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.float32, device='cpu')
        self.dones = torch.zeros((self.buffer_size, self.rollout_length), dtype=torch.uint8, device='cpu')
        self.last_observation = torch.zeros((self.buffer_size, *self.obs_shape), dtype=torch.uint8, device='cpu')
        self.last_done = torch.zeros(self.buffer_size, dtype=torch.uint8, device='cpu')
        if self.lstm_hidden_size > 0:
            self.lstm_states = {
                "hidden": torch.zeros((self.buffer_size, self.rollout_length, self.lstm_hidden_size), dtype=torch.float32, device='cpu'),
                "cell": torch.zeros((self.buffer_size, self.rollout_length, self.lstm_hidden_size), dtype=torch.float32, device='cpu'),
            }
        self.pos = 0
        self.full = False

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        last_observation: torch.Tensor,
        last_done: torch.Tensor,
        lstm_states: Dict[str, torch.Tensor] = None,
    ):
        start_idx = self.pos
        end_idx = (self.pos + self.n_envs) % self.buffer_size

        obs = observations.transpose(0, 1).to('cpu')
        act = actions.transpose(0, 1).to('cpu')
        rew = rewards.transpose(0, 1).to('cpu')
        don = dones.transpose(0, 1).to('cpu')

        if lstm_states is not None:
            lstm_sts_hidden = lstm_states["hidden"].transpose(0, 1).to('cpu')
            lstm_sts_cell = lstm_states["cell"].transpose(0, 1).to('cpu')

        # Handle circular buffer logic
        if end_idx > start_idx:
            self.observations[start_idx:end_idx] = obs
            self.actions[start_idx:end_idx] = act
            self.rewards[start_idx:end_idx] = rew
            self.dones[start_idx:end_idx] = don
            self.last_observation[start_idx:end_idx] = last_observation
            self.last_done[start_idx:end_idx] = last_done
            if self.lstm_hidden_size > 0:
                self.lstm_states["hidden"][start_idx:end_idx] = lstm_sts_hidden
                self.lstm_states["cell"][start_idx:end_idx] = lstm_sts_cell
        else:
            split_point = self.buffer_size - start_idx
            self.observations[start_idx:] = obs[:split_point]
            self.observations[:end_idx] = obs[split_point:]
            self.actions[start_idx:] = act[:split_point]
            self.actions[:end_idx] = act[split_point:]
            self.rewards[start_idx:] = rew[:split_point]
            self.rewards[:end_idx] = rew[split_point:]
            self.dones[start_idx:] = don[:split_point]
            self.dones[:end_idx] = don[split_point:]
            self.last_observation[start_idx:] = last_observation[:split_point]
            self.last_observation[:end_idx] = last_observation[split_point:]
            self.last_done[start_idx:] = last_done[:split_point]
            self.last_done[:end_idx] = last_done[split_point:]
            
            if self.lstm_hidden_size > 0:
                self.lstm_states["hidden"][start_idx:] = lstm_sts_hidden[:split_point]
                self.lstm_states["hidden"][:end_idx] = lstm_sts_hidden[split_point:]
                self.lstm_states["cell"][start_idx:] = lstm_sts_cell[:split_point]
                self.lstm_states["cell"][:end_idx] = lstm_sts_cell[split_point:]

        self.pos = end_idx
        self.full = self.full or end_idx < start_idx

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.async_:
            self.data_ready_event.wait()
            self.lock.acquire()
            data = self._sample(batch_size)
            self.lock.release()
            return data
        else:
            return self._sample(batch_size)
        
    def _sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        upper_bound = self.buffer_size if self.full else self.pos
        indices = random.sample(range(upper_bound), batch_size)
        return self._get_samples(indices)

    def _get_samples(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "last_observation": self.last_observation[indices],
            "last_done": self.last_done[indices],
            "indices": indices,
        }
        
        if self.lstm_hidden_size > 0:
            data["lstm_states_hidden"] = self.lstm_states["hidden"][indices]
            data["lstm_states_cell"] = self.lstm_states["cell"][indices]
            
        data = {k: v.to(self.device) for k, v in data.items() if k != "indices"}
        data["indices"] = indices
        return data
        
class SingleRolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        lstm_hidden_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        q_lambda: float = 0.65,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.rollout_length = rollout_length
        self.n_envs = n_envs
        self.lstm_hidden_size = lstm_hidden_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.q_lambda = q_lambda
        self.gamma = gamma
        self.device = device
        self.lock = threading.Lock()
        self.init_buffers()
        
    def init_buffers(self) -> None:
        self.observations = torch.zeros(
            (self.rollout_length, self.n_envs, *self.obs_shape),
            dtype=torch.uint8,
            device="cpu"
        )
        self.actions = torch.zeros(
            (self.rollout_length, self.n_envs),
            dtype=torch.uint8,
            device="cpu"
        )
        self.rewards = torch.zeros(
            (self.rollout_length, self.n_envs),
            dtype=torch.float32,
            device="cpu"
        )
        self.dones = torch.zeros(
            (self.rollout_length, self.n_envs),
            dtype=torch.uint8,
            device="cpu"
        )
        
        if self.lstm_hidden_size > 0:
            self.lstm_states = {
                "hidden": torch.zeros(
                    (self.rollout_length, self.n_envs, self.lstm_hidden_size),
                    dtype=torch.float32,
                    device="cpu"
                ),
                "cell": torch.zeros(
                    (self.rollout_length, self.n_envs, self.lstm_hidden_size),
                    dtype=torch.float32,
                    device="cpu"
                ),
            }
        self.pos = 0
        self.full = False

    def reset_half(self) -> None:
        self.observations[:self.rollout_length // 2] = self.observations[self.rollout_length // 2:]
        self.actions[:self.rollout_length // 2] = self.actions[self.rollout_length // 2:]
        self.rewards[:self.rollout_length // 2] = self.rewards[self.rollout_length // 2:]
        self.dones[:self.rollout_length // 2] = self.dones[self.rollout_length // 2:]
        
        if self.lstm_hidden_size > 0:
            self.lstm_states["hidden"][:self.rollout_length // 2] = self.lstm_states["hidden"][self.rollout_length // 2:]
            self.lstm_states["cell"][:self.rollout_length // 2] = self.lstm_states["cell"][self.rollout_length // 2:]
            
        self.pos = self.rollout_length // 2
        self.full = False

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        lstm_states: Dict[str, torch.Tensor] = None,
    ):
        self.observations[self.pos] = torch.from_numpy(observations).to("cpu", dtype=torch.uint8)
        self.actions[self.pos] = torch.from_numpy(actions).to("cpu", dtype=torch.uint8)
        self.rewards[self.pos] = torch.from_numpy(rewards).to("cpu", dtype=torch.float32)
        self.dones[self.pos] = torch.from_numpy(dones).to("cpu", dtype=torch.uint8)
        
        if self.lstm_hidden_size > 0:
            self.lstm_states["hidden"][self.pos] = lstm_states[0].clone().cpu()
            self.lstm_states["cell"][self.pos] = lstm_states[1].clone().cpu()
        
        self.pos += 1
        if self.pos == self.rollout_length:
            self.full = True
