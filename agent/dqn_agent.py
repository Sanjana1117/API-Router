from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn import DQN


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, state_size=13, action_size=3):

        self.state_size = state_size
        self.action_size = action_size
        self.target_update_frequency = 10
        self.train_steps = 0

        # Neural network
        self.policy_net = DQN(state_size, action_size)

        # Target network
        self.target_net = DQN(state_size, action_size)

        # Start with same weights
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=0.001
        )

        # Replay memory
        self.memory = ReplayBuffer()

        # Discount factor
        self.gamma = 0.99

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def choose_action(self, state, action_mask=None):

    # EXPLORATION
        if random.random() < self.epsilon:

            if action_mask is None:
                return random.randrange(self.action_size)

            valid_actions = [
                i
                for i in range(self.action_size)
                if action_mask[i]
            ]

            # Safety check
            if len(valid_actions) == 0:
                return random.randrange(self.action_size)

            return random.choice(valid_actions)

        # EXPLOITATION
        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            q_values = self.policy_net(state)

        # ACTION MASKING
        if action_mask is not None:

            for i in range(self.action_size):
                if not action_mask[i]:
                    q_values[i] = float("-inf")

        return torch.argmax(q_values).item()

    def train(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # Q-value for the action we actually took
        current_q_values = self.policy_net(states)

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Future Q-values
        with torch.no_grad():

            next_q_values = self.target_net(next_states)

            max_next_q_values = next_q_values.max(
                dim=1
            )[0]

            target_q_values = rewards + (
                self.gamma *
                max_next_q_values *
                (1 - dones)
            )

        # Calculate loss
        loss = nn.SmoothL1Loss()(
            current_q_values,
            target_q_values
        )

        # Update network
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_frequency == 0:

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # Reduce exploration
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

      