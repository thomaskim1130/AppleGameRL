import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pygame
import time

class Actor(nn.Module):
    def __init__(self, height, width):
        super(Actor, self).__init__()
        # Convolutional layers to process the grid
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc = nn.Linear(64 * height * width, 128)
        # Output heads for each action dimension
        self.x1_head = nn.Linear(128, width)
        self.y1_head = nn.Linear(128, height)
        self.rect_width_head = nn.Linear(128, width)
        self.rect_height_head = nn.Linear(128, height)

    def forward(self, grid):
        # Input grid shape: (batch_size, height, width)
        x = grid.unsqueeze(1)  # Add channel: (batch_size, 1, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64 * height * width)
        x = F.relu(self.fc(x))
        # Output logits for each action component
        x1_logits = self.x1_head(x)
        y1_logits = self.y1_head(x)
        rect_width_logits = self.rect_width_head(x)
        rect_height_logits = self.rect_height_head(x)
        return x1_logits, y1_logits, rect_width_logits, rect_height_logits

    def sample_action(self, grid):
        x1_logits, y1_logits, rect_width_logits, rect_height_logits = self.forward(grid)
        # Create categorical distributions
        x1_dist = Categorical(logits=x1_logits)
        y1_dist = Categorical(logits=y1_logits)
        rect_width_dist = Categorical(logits=rect_width_logits)
        rect_height_dist = Categorical(logits=rect_height_logits)
        # Sample actions
        x1 = x1_dist.sample()
        y1 = y1_dist.sample()
        rect_width = rect_width_dist.sample()
        rect_height = rect_height_dist.sample()
        # Combine into action tensor
        action = torch.stack([x1, y1, rect_width, rect_height], dim=1)
        # Compute total log probability
        log_prob = x1_dist.log_prob(x1) + y1_dist.log_prob(y1) + \
                   rect_width_dist.log_prob(rect_width) + rect_height_dist.log_prob(rect_height)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, height, width, embedding_dim=8):
        super(Critic, self).__init__()
        # Process grid
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_grid = nn.Linear(64 * height * width, 128)
        # Embed action components
        self.x1_embed = nn.Embedding(width, embedding_dim)
        self.y1_embed = nn.Embedding(height, embedding_dim)
        self.rect_width_embed = nn.Embedding(width, embedding_dim)
        self.rect_height_embed = nn.Embedding(height, embedding_dim)
        # Combine features
        self.fc = nn.Linear(128 + 4 * embedding_dim, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, grid, action):
        # Grid shape: (batch_size, height, width)
        x = grid.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        grid_features = F.relu(self.fc_grid(x))
        # Action shape: (batch_size, 4)
        x1_emb = self.x1_embed(action[:, 0])
        y1_emb = self.y1_embed(action[:, 1])
        rect_width_emb = self.rect_width_embed(action[:, 2])
        rect_height_emb = self.rect_height_embed(action[:, 3])
        action_features = torch.cat([x1_emb, y1_emb, rect_width_emb, rect_height_emb], dim=1)
        # Combine and process
        features = torch.cat([grid_features, action_features], dim=1)
        x = F.relu(self.fc(features))
        q_value = self.out(x)
        return q_value

class Value(nn.Module):
    def __init__(self, height, width):
        super(Value, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * height * width, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, grid):
        x = grid.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        v_value = self.out(x)
        return v_value

class QACAgent:
    def __init__(self, env, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3, lr_value=1e-3):
        """Initialize the QAC agent with environment and hyperparameters."""
        self.env = env
        self.gamma = gamma
        self.actor = Actor(env.height, env.width)
        self.critic = Critic(env.height, env.width)
        self.value = Value(env.height, env.width)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)

    def train(self, num_episodes, verbose=True):
        """Train the agent over a specified number of episodes."""
        print("Starting training...")
        for episode in range(num_episodes):
            print(f"Starting episode {episode + 1}")
            state = self.env.reset()[0]  # Get initial grid
            state = torch.tensor(state, dtype=torch.float32) / 9.0  # Normalize to [0,1]
            done = False
            transitions = []
            step = 0

            while not done:
                step += 1
                action, log_prob = self.actor.sample_action(state.unsqueeze(0))
                action = action.squeeze(0)  # Shape: (4,)
                action_np = action.cpu().numpy()
                next_state, reward, done, _, _ = self.env.step(action_np)
                next_state = torch.tensor(next_state, dtype=torch.float32) / 9.0
                transitions.append((state, action, log_prob, reward, next_state, done))
                state = next_state
                self.env.render()
                pygame.event.pump()
                time.sleep(0.1)  # Slow down to make rendering visible
                print(f"Step {step}, Action: {action_np}, Reward: {reward}")

            # Process collected transitions
            states, actions, log_probs, rewards, next_states, dones = zip(*transitions)
            states = torch.stack(states)  # (ep_len, height, width)
            actions = torch.stack(actions)  # (ep_len, 4)
            log_probs = torch.stack(log_probs)  # (ep_len,)
            rewards = torch.tensor(rewards, dtype=torch.float32)  # (ep_len,)
            next_states = torch.stack(next_states)  # (ep_len, height, width)
            dones = torch.tensor(dones, dtype=torch.float32)  # (ep_len,)

            # Compute targets and losses
            with torch.no_grad():
                V_next = self.value(next_states).squeeze()  # (ep_len,)
            V_target = rewards + self.gamma * V_next * (1 - dones)  # TD target
            Q = self.critic(states, actions).squeeze()  # (ep_len,)
            V = self.value(states).squeeze()  # (ep_len,)
            advantage = Q - V  # Advantage estimate

            # Compute losses
            loss_critic = ((Q - V_target) ** 2).mean()
            loss_value = ((V - V_target) ** 2).mean()
            loss_actor = - (log_probs * advantage.detach()).mean()

            # Update networks
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            episode_reward = sum(rewards)
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}, "
                      f"Actor Loss: {loss_actor.item():.4f}, Critic Loss: {loss_critic.item():.4f}, "
                      f"Value Loss: {loss_value.item():.4f}")

# Example usage:
if __name__ == "__main__":
    from AppleGame import AppleGameEnv
    import pygame
    import time
    
    if hasattr(torch.backends, 'nnpack'):
        torch.backends.nnpack.enabled = False
        
    env = AppleGameEnv(width=17, height=10, render_mode='human')
    agent = QACAgent(env)
    try:
        agent.train(num_episodes=100, verbose=True)
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        env.close()