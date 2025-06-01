import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# AppleGameEnv이 아래 import 경로에 있다고 가정합니다.
# 실제 파일 경로에 맞춰 조정하세요.
from AppleGame import AppleGameEnv


class DuelingDQNNetwork(nn.Module):
    def __init__(self, height, width, max_actions=1000):
        super(DuelingDQNNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Shared fully-connected layer
        self.fc_shared = nn.Linear(64 * height * width, 128)
        # Value stream
        self.fc_value = nn.Linear(128, 1)
        # Advantage stream
        self.fc_adv = nn.Linear(128, max_actions)

        # 가중치 초기화 (Xavier)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc_shared.weight)
        nn.init.xavier_uniform_(self.fc_value.weight)
        nn.init.xavier_uniform_(self.fc_adv.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc_shared.bias)
        nn.init.zeros_(self.fc_value.bias)
        nn.init.zeros_(self.fc_adv.bias)

    def forward(self, state, num_actions):
        """
        state: Tensor of shape (batch_size, height, width), values are normalized [0, 1]
        num_actions: int, number of valid actions in this batch
        """
        x = state.unsqueeze(1)              # (batch_size, 1, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)           # flatten
        x = F.relu(self.fc_shared(x))

        # Value and Advantage streams
        value = self.fc_value(x)            # (batch_size, 1)
        adv = self.fc_adv(x)                # (batch_size, max_actions)
        adv_mean = adv.mean(dim=1, keepdim=True)  # (batch_size, 1)
        q_values = value + (adv - adv_mean)       # (batch_size, max_actions)

        # Mask out invalid actions (set Q to -inf if action >= num_actions)
        device = q_values.device
        mask = torch.arange(q_values.size(-1), device=device).unsqueeze(0) < num_actions
        neg_inf = torch.full_like(q_values, -float('inf'))
        q_values = torch.where(mask, q_values, neg_inf)

        return q_values  # (batch_size, max_actions)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_num_actions):
        """
        Append a single transition.
        state, next_state: numpy arrays (height, width), normalized
        action: int
        reward: float
        done: bool
        next_num_actions: int (# of valid actions in next_state)
        """
        self.buffer.append((state, action, reward, next_state, done, next_num_actions))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    def __init__(
        self,
        env: AppleGameEnv,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=100000,
        min_buffer_size=1000,
        update_freq=4,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        max_actions=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_actions = max_actions

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network & Target network
        self.policy_net = DuelingDQNNetwork(env.height, env.width, max_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(env.height, env.width, max_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # TensorBoard logger
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("runs", "DuelingDQN_AppleGame", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = log_dir

        self.total_steps = 0

    def select_action(self, state, num_actions):
        """
        ε-greedy action selection (training mode).
        state: numpy array (height, width), normalized [0,1]
        num_actions: int
        """
        if num_actions == 0:
            return None

        if random.random() < self.epsilon:
            return random.randrange(num_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, num_actions)  # (1, max_actions)
            action = q_values.argmax(dim=1).item()
            return action

    def select_action_fixed_epsilon(self, state, num_actions, fixed_epsilon):
        """
        ε-greedy action selection (inference mode, fixed ε).
        state: numpy array (height, width), normalized [0,1]
        num_actions: int
        fixed_epsilon: float in [0, 1]
        """
        if num_actions == 0:
            return None

        if random.random() < fixed_epsilon:
            return random.randrange(num_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, num_actions)
            action = q_values.argmax(dim=1).item()
            return action

    def update_model(self):
        """
        Sample a batch from replay buffer and update policy network.
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_num_actions_list = zip(*batch)

        # Convert to Tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device) / 9.0
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device) / 9.0
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_num_actions_tensor = torch.tensor(next_num_actions_list, dtype=torch.long, device=self.device)

        # Compute target Q using target_net
        with torch.no_grad():
            next_q_all = self.target_net(next_states_tensor, self.max_actions)  # (batch, max_actions)
            next_max_q_list = []
            for i in range(self.batch_size):
                n_actions = next_num_actions_tensor[i].item()
                if n_actions > 0:
                    next_max_q = next_q_all[i, :n_actions].max()
                else:
                    next_max_q = torch.tensor(0.0, device=self.device)
                next_max_q_list.append(next_max_q)
            next_max_q_tensor = torch.stack(next_max_q_list)  # (batch,)
            target_q = rewards_tensor + self.gamma * next_max_q_tensor * (1 - dones_tensor)

        # Compute current Q with policy_net
        q_all = self.policy_net(states_tensor, self.max_actions)  # (batch, max_actions)
        current_q = q_all.gather(dim=1, index=actions_tensor.unsqueeze(1)).squeeze(1)  # (batch,)

        # MSE loss
        loss = F.mse_loss(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=1000, verbose=True):
        """
        Dueling DQN training loop.
        """
        print("=== Dueling DQN Training Start ===")
        pbar = trange(1, num_episodes + 1, desc="Episodes", ncols=80)
        for episode in pbar:
            state, info = self.env.reset()
            state = state.astype(np.float32) / 9.0
            state = state.copy()
            feasible_actions = info['feasible_actions']
            num_actions = len(feasible_actions)

            episode_reward = 0.0
            done = False

            # If no valid actions at start, skip
            if num_actions == 0:
                if verbose:
                    pbar.set_postfix_str(f"Ep {episode}: No initial actions, skip")
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                self.writer.add_scalar('Score/Episode', episode_reward, episode)
                self.writer.add_scalar('Epsilon/Episode', self.epsilon, episode)
                continue

            while not done:
                if num_actions == 0:
                    done = True
                    break

                action = self.select_action(state, num_actions)
                if action is None:
                    done = True
                    break

                next_state, reward, done, _, info = self.env.step(action)
                next_state = next_state.astype(np.float32) / 9.0
                next_state = next_state.copy()
                next_feasible_actions = info['feasible_actions']
                next_num_actions = len(next_feasible_actions)

                episode_reward = info['score']

                self.replay_buffer.push(state, action, reward, next_state, done, next_num_actions)

                state = next_state
                num_actions = next_num_actions

                self.total_steps += 1
                if self.total_steps % self.update_freq == 0:
                    loss = self.update_model()
                    if loss is not None:
                        self.writer.add_scalar('Loss/Step', loss, self.total_steps)

                # (선택 사항) render
                self.env.render()

            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # TensorBoard logging per episode
            self.writer.add_scalar('Score/Episode', episode_reward, episode)
            self.writer.add_scalar('Epsilon/Episode', self.epsilon, episode)

            if verbose:
                pbar.set_postfix({
                    "Score": f"{episode_reward:.2f}",
                    "Epsilon": f"{self.epsilon:.3f}"
                })

        print("=== Dueling DQN Training Finished ===")
        self.writer.close()

        checkpoint_path = os.path.join(self.checkpoint_dir, "dueling_dqn_apple_game.pth")
        torch.save(self.policy_net.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    def close(self):
        self.env.close()


if __name__ == "__main__":
    import pygame

    # 1) Create environment
    env = AppleGameEnv(width=17, height=10, render_mode='human', max_actions=1000)

    # 2) Instantiate agent
    agent = DuelingDQNAgent(
        env,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=100000,
        min_buffer_size=1000,
        update_freq=4,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        max_actions=1000,
    )

    # 3) (Optional) Load a saved checkpoint before evaluation
    # checkpoint_path = "runs/DuelingDQN_AppleGame/YYYYMMDD_HHMMSS/dueling_dqn_apple_game.pth"
    # agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    # agent.target_net.load_state_dict(agent.policy_net.state_dict())
    # agent.policy_net.eval()
    # agent.target_net.eval()

    # -----------------------------------------------------------
    # 4) Training (uncomment if you want to train now)
    # -----------------------------------------------------------
    # agent.train(num_episodes=1000, verbose=True)
    # agent.close()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # 5) Inference / Evaluation Loop
    # -----------------------------------------------------------
    fixed_epsilon = 0.05
    num_eval_episodes = 20  # Evaluate over 20 episodes

    all_scores = []
    for ep in range(1, num_eval_episodes + 1):
        state, info = env.reset()
        state = state.astype(np.float32) / 9.0
        state = state.copy()
        feasible_actions = info['feasible_actions']
        num_actions = len(feasible_actions)

        done = False
        episode_reward = 0.0

        while not done:
            if num_actions == 0:
                break

            action = agent.select_action_fixed_epsilon(state, num_actions, fixed_epsilon)
            if action is None:
                break

            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.astype(np.float32) / 9.0
            next_state = next_state.copy()
            feasible_actions = info['feasible_actions']
            num_actions = len(feasible_actions)

            episode_reward = info['score']
            state = next_state

            # (선택 사항) render
            env.render()

        all_scores.append(episode_reward)
        print(f"Eval Episode {ep}/{num_eval_episodes} - Score: {episode_reward:.2f}")

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n=== Evaluation Finished over {num_eval_episodes} episodes ===")
    print(f"Average Score: {avg_score:.2f}")

    agent.close()
