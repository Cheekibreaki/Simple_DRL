import random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class MultiRobotExplorationEnv:
    def __init__(self, grid_size=40, num_robots=2, sensor_range=1, max_steps=100, obstacle_density=60):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.sensor_range = sensor_range
        self.state_space = spaces.Box(low=0, high=1, shape=(num_robots, grid_size, grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.seed()
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.gridworld = np.zeros((self.grid_size, self.grid_size))
        self.robots_positions = [self.np_random.randint(self.grid_size, size=(2)) for _ in range(self.num_robots)]
        self.steps_taken = 0
        self.obstacles = self.add_obstacles()
        return self._observe()

    def add_obstacles(self):
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density / 100)
        return [self.np_random.randint(self.grid_size, size=(2)) for _ in range(num_obstacles)]

    def step(self, actions):
        assert len(actions) == self.num_robots
        rewards = []
        for i in range(self.num_robots):
            action = actions[i]
            next_position = self._get_next_position(self.robots_positions[i], action)
            next_position = self._clip_position(next_position)
            if self._is_obstacle(next_position):  # Check if the next position is an obstacle
                next_position = self.robots_positions[i]  # Keep the agent's position unchanged
            self.robots_positions[i] = next_position
            rewards.append(self._get_reward(next_position))
        self.steps_taken += 1
        done = np.sum(self.gridworld) == self.grid_size * self.grid_size

        if self.steps_taken >= self.max_steps:
            done = True
            rewards = [0 for _ in range(self.num_robots)]  # No reward if not completed within max steps

        if done:
            rewards = [100 for _ in range(self.num_robots)]  # Joint reward for completing the task

        return self._observe(), rewards, done, {}

    def _get_next_position(self, position, action):
        if action == 0:  # Move up
            return position + np.array([-1, 0])
        elif action == 1:  # Move down
            return position + np.array([1, 0])
        elif action == 2:  # Move left
            return position + np.array([0, -1])
        elif action == 3:  # Move right
            return position + np.array([0, 1])
        return position

    def _is_obstacle(self, position):
        obstacles_set = {tuple(obs) for obs in self.obstacles}
        position_tuple = tuple(position)
        return position_tuple in obstacles_set

    def _clip_position(self, position):
        return np.clip(position, 0, self.grid_size - 1)

    def _get_reward(self, position):
        if self.gridworld[position[0], position[1]] == 0:
            self.gridworld[position[0], position[1]] = 1
            return 1
        return 0

    def _observe(self):
        obs = np.zeros((self.num_robots, self.grid_size, self.grid_size))
        for i in range(self.num_robots):
            x, y = self.robots_positions[i]
            obs[i, max(0, x - self.sensor_range):min(self.grid_size, x + self.sensor_range + 1),
            max(0, y - self.sensor_range):min(self.grid_size, y + self.sensor_range + 1)] = 1
            # Hide states behind obstacles
            for obs_pos in self.obstacles:
                if obs_pos[0] >= max(0, x - self.sensor_range) and obs_pos[0] < min(self.grid_size,
                                                                                    x + self.sensor_range + 1) \
                        and obs_pos[1] >= max(0, y - self.sensor_range) and obs_pos[1] < min(self.grid_size,
                                                                                             y + self.sensor_range + 1):
                    obs[i, obs_pos[0], obs_pos[1]] = 0
        return obs

    def print_gridworld(self):
        grid_copy = np.copy(self.gridworld)
        for x, y in self.robots_positions:
            grid_copy[x, y] = 2  # Mark agents with 2
        for x, y in self.obstacles:
            grid_copy[x, y] = 3  # Mark obstacles with 3
        print(grid_copy)


2]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(np.prod(self.state_size), 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def act(self, state):
        if isinstance(state, list):  # Multiple agents' states
            if np.random.rand() <= self.epsilon:
                return [random.randrange(self.action_size[i]) for i in range(len(state))]
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor.view(1, -1))
            return [torch.argmax(q_value).item() for q_value in q_values]
        else:  # Single agent's state
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor.view(1, -1))
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        for agent_memory in self.memory:
            if len(agent_memory) < batch_size:
                return
            minibatch = random.sample(agent_memory, batch_size)
            state_batch = torch.tensor([state for state, _, _, _, _ in minibatch], dtype=torch.float32).to(self.device)
            action_batch = torch.tensor([action for _, action, _, _, _ in minibatch], dtype=torch.long).to(self.device)
            reward_batch = torch.tensor([reward for _, _, reward, _, _ in minibatch], dtype=torch.float32).to(
                self.device)
            next_state_batch = torch.tensor([next_state for _, _, _, next_state, _ in minibatch],
                                            dtype=torch.float32).to(self.device)
            done_batch = torch.tensor([done for _, _, _, _, done in minibatch], dtype=torch.float32).to(self.device)

            q_values = self.model(state_batch)
            next_q_values = self.model(next_state_batch)
            q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)

            loss = self.loss_fn(q_value, expected_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._decay_epsilon()

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    grid_size = 10
    num_robots = 2
    sensor_range = 1
    obstacle_density = 60  # 0-100

    env = MultiRobotExplorationEnv(grid_size, num_robots, sensor_range, obstacle_density)
    state_size = num_robots * (2 * sensor_range + 1) ** 2
    action_size = 4  # Four possible actions: Up, Down, Left, Right

    agents = [DQNAgent(state_size, action_size) for _ in range(num_robots)]

    num_episodes = 10000
    batch_size = 32

    env.print_gridworld()
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        total_steps = 0
        while not done:
            actions = [agent.act(state[i]) for i, agent in enumerate(agents)]
            next_state, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(state[i], actions[i], rewards[i], next_state[i], done)

            state = next_state
            total_reward += sum(rewards)
            total_steps += 1

        for agent in agents:
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward}, Total steps: {total_steps} ")


# # Execution
# state = env.reset()
# env.print_gridworld()
# done = False
# total_steps = 0
# total_reward = 0
# while not done:
#         actions = [agent.act(state) for agent in agents]
#         next_state, rewards, done, _ = env.step(actions)
#         state = next_state
#         total_reward += sum(rewards)
#         total_steps += 1
#
# print("Total reward: {total_reward}, Total steps: {total_steps} ")