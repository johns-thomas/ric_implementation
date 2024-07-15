import gym
from gym import spaces
import numpy as np
class SyntheticLambdaEnv(gym.Env):
    def __init__(self, synthetic_data):
        super(SyntheticLambdaEnv, self).__init__()
        self.synthetic_data = synthetic_data
        self.current_index = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0=decrease, 1=maintain, 2=increase
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self):
        self.current_index = 0
        self.state = self._get_state(self.current_index)
        return self.state

    def step(self, action):
        self._apply_action(action)
        self.current_index += 1
        
        if self.current_index >= len(self.synthetic_data):
            self.current_index = len(self.synthetic_data) - 1

        new_state = self._get_state(self.current_index)
        reward = self._calculate_reward(new_state)
        done = self.current_index == len(self.synthetic_data) - 1
        return new_state, reward, done, {}

    def _get_state(self, index):
        data_point = self.synthetic_data.iloc[index]
        state = [
            data_point['memory_allocation'],
            data_point['timeout_setting'],
            data_point['concurrency_level'],
            data_point['average_execution_time'],
            data_point['error_rate'],
            data_point['cold_start_frequency'],
            data_point['cost']
        ]
        return np.array(state, dtype=np.float32)

    def _apply_action(self, action):
        # Simulate the effect of the action on the current configuration
        current_data_point = self.synthetic_data.iloc[self.current_index]
        if action == 0:
            current_data_point['memory_allocation'] = max(128, current_data_point['memory_allocation'] - 128)
        elif action == 2:
            current_data_point['memory_allocation'] = min(2048, current_data_point['memory_allocation'] + 128)
        # No change for action == 1 (maintain)

    def _calculate_reward(self, state):
        execution_time, error_rate, cold_start_frequency, cost = state[3], state[4], state[5], state[6]
        performance_reward = 1.0 / execution_time
        error_penalty = 10.0 * error_rate
        cold_start_penalty = 5.0 * cold_start_frequency
        cost_penalty = 1.0 * cost
        return performance_reward - error_penalty - cold_start_penalty - cost_penalty

#synthetic_env = SyntheticLambdaEnv(synthetic_data)
