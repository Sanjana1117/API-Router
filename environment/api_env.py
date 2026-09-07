import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.providers import providers


class APIRoutingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.step_count = 0
        self.reset_interval = 20


        # We have 3 possible actions:
        # 0 -> Provider A
        # 1 -> Provider B
        # 2 -> Provider C
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        # Reset all providers
        for provider in providers:
            provider.reset()

        # Start with normal traffic
        traffic = np.random.randint(1, 10)

        state = self._get_state(traffic)

        return state, {}

    def _get_state(self, traffic):

        state = [
            traffic,

            providers[0].remaining_requests,
            providers[1].remaining_requests,
            providers[2].remaining_requests,

            providers[0].latency,
            providers[1].latency,
            providers[2].latency,

            providers[0].cost,
            providers[1].cost,
            providers[2].cost,

            providers[0].success_rate,
            providers[1].success_rate,
            providers[2].success_rate
        ]

        return np.array(state, dtype=np.float32)

    def step(self, action):

        self.step_count += 1

        if self.step_count % self.reset_interval == 0:

            for provider in providers:
                provider.reset()

        # Select provider
        selected_provider = providers[action]

        # Send request
        result = selected_provider.handle_request()

        if result["status_code"] == 429:

            reward = -20

        elif result["success"]:

            reward = 10

            # Penalize high cost
            reward -= result["cost"]

            # Penalize high latency
            reward -= result["latency"] / 100

        else:

            reward = -10

        traffic = np.random.randint(1, 10)

        next_state = self._get_state(traffic)

        terminated = False
        truncated = False

        return next_state, reward, terminated, truncated, result    

    def get_action_mask(self):

        return np.array([
            providers[0].remaining_requests > 0,
            providers[1].remaining_requests > 0,
            providers[2].remaining_requests > 0
        ])