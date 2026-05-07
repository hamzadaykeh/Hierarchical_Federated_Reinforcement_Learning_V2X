import numpy as np
import gymnasium as gym
from gymnasium import spaces
#pip install numpy gymnasium matplotlib


class HighwayEnv(gym.Env):
    def __init__(self, num_vehicles=5, dt=0.1):
        super().__init__()

        self.num_vehicles = num_vehicles
        self.dt = dt
        self.current_step = 0

        # RL controls only vehicle 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(1,),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.positions = np.linspace(0, 50, self.num_vehicles)
        self.velocities = np.ones(self.num_vehicles) * 20.0
        self.accelerations = np.zeros(self.num_vehicles)
        self.lanes = np.zeros(self.num_vehicles)
        self.current_step = 0
        self.max_steps = 200

        return self._get_obs(), {}

    def step(self, action):
        # RL action for vehicle 0
        self.accelerations[0] = action[0]

         # Other vehicles follow ODE car-following
        for i in range(1, self.num_vehicles):
            distance = self.positions[i-1] - self.positions[i]
            desired_gap = 10.0

            self.accelerations[i] = 0.5 * (20.0 - self.velocities[i]) \
                                + 0.2 * (distance - desired_gap)

        # Physics update
        for i in range(self.num_vehicles):
            x = self.positions[i]
            v = self.velocities[i]
            a = self.accelerations[i]

            # ODE:
            # dx/dt = v
            # dv/dt = a

            dx = v
            dv = a

            x_new = x + dx * self.dt
            v_new = v + dv * self.dt

            self.positions[i] = x_new
            self.velocities[i] = v_new
        obs = self._get_obs()

        reward = self._compute_reward()
        distance_to_front = self.positions[1] - self.positions[0]

        terminated = False
        truncated = False
        if self.positions[1] - self.positions[0] < 5:
            terminated = True
        # Collision condition
        if distance_to_front < 2.0:
            terminated = True

        # Max episode length
        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Observation of vehicle 0
        return np.array([
            self.positions[0],
            self.velocities[0],
            self.accelerations[0],
            self.positions[1] - self.positions[0]  # distance to front car
        ], dtype=np.float32)

    def _compute_reward(self):
        distance_to_front = self.positions[1] - self.positions[0]

        speed_error = abs(self.velocities[0] - 20.0)

        reward = 0

        # speed objective
        reward += 1.0 - (speed_error / 10)

        # smoothness
        reward += - abs(self.accelerations[0]) * 0.05

        # safety
        if distance_to_front < 5:
            reward -= 3
        elif distance_to_front > 10:
            reward += 1

        return reward