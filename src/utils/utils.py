import gymnasium as gym
import numpy as np

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self):
        observations, info = self.env.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        observations, rewards, terms, truncs, infos = self.env.step(action)
        
        # fix autoreset difference with gymnasium
        ep_done = np.logical_or(terms, truncs)
        env_ids_to_reset = np.where(ep_done)[0]
        if len(env_ids_to_reset) > 0:
            (
                reset_obs,
                _,
                _,
                _,
                _,
            ) = self.env.step(np.zeros_like(action), env_ids_to_reset)
            observations[env_ids_to_reset] = reset_obs

        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            terms,
            truncs,
            infos,
        )

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    if end_e < start_e:
        return max(slope * t + start_e, end_e)
    else:
        return min(slope * t + start_e, end_e)