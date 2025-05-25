import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pygame
from torch.utils.tensorboard import SummaryWriter
import os
import time

class Actor(nn.Module):
    def __init__(self, height, width, max_actions=1000):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * height * width, 128)
        self.action_head = nn.Linear(128, max_actions)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, state, num_actions):
        if torch.isnan(state).any():
            raise ValueError("NaN detected in input state")
        
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.action_head(x)
        
        # Mask logits beyond num_actions
        mask = torch.arange(logits.size(-1), device=logits.device) < num_actions
        logits = torch.where(mask, logits, torch.tensor(-float('inf'), device=logits.device))
        
        if torch.isnan(logits).any():
            raise ValueError("NaN detected in logits")
        return logits

    def sample_action(self, state, num_actions):
        logits = self.forward(state, num_actions)
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        # Ensure log_prob is 1D
        if log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)
        return action_idx, log_prob

class Critic(nn.Module):
    def __init__(self, height, width, embedding_dim=8):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_grid = nn.Linear(64 * height * width, 128)
        self.x1_embed = nn.Embedding(width, embedding_dim)
        self.y1_embed = nn.Embedding(height, embedding_dim)
        self.rect_width_embed = nn.Embedding(width, embedding_dim)
        self.rect_height_embed = nn.Embedding(height, embedding_dim)
        self.fc = nn.Linear(128 + 4 * embedding_dim, 64)
        self.out = nn.Linear(64, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc_grid.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc_grid.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, state, action):
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        grid_features = F.relu(self.fc_grid(x))
        x1_emb = self.x1_embed(action[:, 0])
        y1_emb = self.y1_embed(action[:, 1])
        rect_width_emb = self.rect_width_embed(action[:, 2])
        rect_height_emb = self.rect_height_embed(action[:, 3])
        action_features = torch.cat([x1_emb, y1_emb, rect_width_emb, rect_height_emb], dim=1)
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
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, state):
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        v_value = self.out(x)
        return v_value

class QACAgent:
    def __init__(self, env, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3, lr_value=1e-3):
        self.env = env
        self.gamma = gamma
        self.actor = Actor(env.height, env.width, max_actions=env.max_actions)
        self.critic = Critic(env.height, env.width)
        self.value = Value(env.height, env.width)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        self.episode_rewards = []
        # Initialize TensorBoard writer
        log_dir = f"QAC/apple_game_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, num_episodes, verbose=True, save_path=None):
        print("Starting training...")
        for episode in range(num_episodes):
            print(f"Starting episode {episode + 1}")
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32) / 9.0
            feasible_actions = info['feasible_actions']
            done = False
            transitions = []
            episode_score = 0

            # Check if reset produced no feasible actions
            if not feasible_actions:
                print(f"Episode {episode + 1}: No feasible actions after reset, skipping episode")
                continue

            while not done:
                # Check if feasible actions exist
                if not feasible_actions:
                    print(f"Episode {episode + 1}: No feasible actions, terminating episode")
                    done = True
                    # Store a default transition to maintain learning consistency
                    action_tensor = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                    reward = -0.1  # Match environment's penalty for no feasible actions
                    transitions.append((state, action_tensor, torch.tensor([0.0]), reward, state, True))
                    break

                # Set number of actions
                num_actions = len(feasible_actions)
                print(f"Step {len(transitions) + 1}: num_actions={num_actions}, feasible_actions_length={len(feasible_actions)}")

                # Sample action
                try:
                    action_idx, log_prob = self.actor.sample_action(state.unsqueeze(0), num_actions)
                    action_idx = action_idx.squeeze(0)
                    action_idx_item = action_idx.item()
                except Exception as e:
                    print(f"Error in action sampling: {e}")
                    done = True
                    break

                # Validate action index
                if action_idx_item >= num_actions:
                    print(f"Warning: action_idx {action_idx_item} exceeds num_actions {num_actions}. Clamping to 0")
                    action_idx_item = 0

                print(f"Selected action_idx: {action_idx_item}")

                # Map action index to feasible action tuple
                try:
                    action_np = np.array(feasible_actions[action_idx_item])
                except IndexError as e:
                    print(f"IndexError: {e}, action_idx={action_idx_item}, feasible_actions_length={len(feasible_actions)}")
                    done = True
                    break

                # Step environment
                next_state, reward, env_done, _, info = self.env.step(action_idx_item)
                next_state = torch.tensor(next_state, dtype=torch.float32) / 9.0
                feasible_actions = info['feasible_actions']
                done = env_done

                # Update episode score
                episode_score = info['score']

                # Store transition
                action_tensor = torch.tensor(action_np, dtype=torch.long)
                transitions.append((state, action_tensor, log_prob, reward, next_state, done))
                state = next_state

                self.env.render()
                pygame.event.pump()
                print(f"Step {len(transitions)}, Action Index: {action_idx_item}, Action: {tuple(action_np)}, Reward: {reward}, Score: {episode_score}")

            # Process learning if transitions exist
            if transitions:
                try:
                    states, actions, log_probs, rewards, next_states, dones = zip(*transitions)
                    states = torch.stack(states)
                    actions = torch.stack(actions)
                    log_probs = torch.stack(log_probs)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.stack(next_states)
                    dones = torch.tensor(dones, dtype=torch.float32)

                    with torch.no_grad():
                        V_next = self.value(next_states).squeeze()
                    V_target = rewards + self.gamma * V_next * (1 - dones)
                    Q = self.critic(states, actions).squeeze()
                    V = self.value(states).squeeze()
                    advantage = Q - V

                    # Compute losses
                    loss_critic = F.mse_loss(Q, V_target)
                    loss_value = F.mse_loss(V, V_target)
                    loss_actor = - (log_probs * advantage.detach()).mean()

                    # Optimize critic
                    self.optimizer_critic.zero_grad()
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                    self.optimizer_critic.step()

                    # Optimize value
                    self.optimizer_value.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
                    self.optimizer_value.step()

                    # Optimize actor
                    self.optimizer_actor.zero_grad()
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    self.optimizer_actor.step()

                    episode_reward = rewards.sum().item()
                    self.episode_rewards.append(episode_reward)
                except Exception as e:
                    print(f"Error processing transitions: {e}")
                    print("Log prob shapes:", [lp.shape for lp in log_probs])
                    continue
            else:
                episode_reward = 0
                loss_actor = loss_critic = loss_value = 0.0

            # Log to TensorBoard
            self.writer.add_scalar('Score/Episode', episode_score, episode)
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)
            self.writer.add_scalar('Loss/Actor', loss_actor.item() if isinstance(loss_actor, torch.Tensor) else loss_actor, episode)
            self.writer.add_scalar('Loss/Critic', loss_critic.item() if isinstance(loss_critic, torch.Tensor) else loss_critic, episode)
            self.writer.add_scalar('Loss/Value', loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value, episode)

            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}, Score: {episode_score}, Total Reward: {episode_reward}, "
                      f"Actor Loss: {loss_actor:.4f}, Critic Loss: {loss_critic:.4f}, Value Loss: {loss_value:.4f}")

        self.writer.close()

        if save_path:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'value': self.value.state_dict()
            }, save_path)

if __name__ == "__main__":
    from AppleGame import AppleGameEnv
    import pygame
    import time
    
    if hasattr(torch.backends, 'nnpack'):
        torch.backends.nnpack.enabled = False
        
    env = AppleGameEnv(width=17, height=10, render_mode='human', max_actions=1000)
    agent = QACAgent(env)
    try:
        agent.train(num_episodes=2000, verbose=True, save_path='QAC.pth')
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        env.close()