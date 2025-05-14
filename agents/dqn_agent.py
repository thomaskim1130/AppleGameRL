import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample_w_prob(self, batch_size, prob=None, device='cpu'):
        if len(self.buffer) < batch_size:
            indices = np.arange(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        samples = [self.buffer[i] for i in indices]
        batch = list(zip(*samples))
        states = torch.FloatTensor(np.array(batch[0])).to(device)
        actions = torch.LongTensor(np.array(batch[1])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(batch[2])).to(device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(device)
        dones = torch.FloatTensor(np.array(batch[4])).to(device)
        return (states, actions, rewards, next_states, dones), indices

    def sample(self, batch_size):
        sarsd, indices = self.sample_w_prob(batch_size, None)
        return sarsd, indices, None

    def __len__(self):
        return len(self.buffer)


class MLPQNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN agent for a MultiDiscrete action space,
    with a replay buffer warm-up period before learning starts.
    """
    def __init__(self, env,
                 lr=1e-3,
                 hidden_dim=256,
                 gamma=0.99,
                 buffer_size=10000,
                 batch_size=32,
                 learning_starts=50000,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=5000,
                 target_update_freq=1000,
                 device='cpu'):
        obs_shape = env.observation_space.shape
        self.nvec = env.action_space.nvec.tolist()
        n_actions = int(np.sum(self.nvec))

        # Replay buffer and warm-up threshold
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        # Q-networks
        flat_dim = int(np.prod(obs_shape))
        self.qnet = MLPQNet(flat_dim, hidden_dim, n_actions).to(device)
        self.target = MLPQNet(flat_dim, hidden_dim, n_actions).to(device)
        self.target.load_state_dict(self.qnet.state_dict())

        # Optimizer
        self.opt = optim.AdamW(self.qnet.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Bookkeeping
        self.step_count = 0      # counts env steps / actions
        self.action_space = env.action_space
        self.device = device

    def get_action(self, state, is_training=True):
        self.step_count += 1
        # Exponential epsilon decay
        if is_training:
            self.epsilon *= (1.0 - (1.0 - self.eps_end) / self.eps_decay)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            qvals = self.qnet(x)  # [1, sum(nvec)]
            splits = torch.split(qvals, self.nvec, dim=1)
            actions = [chunk.argmax(dim=1).item() for chunk in splits]
            return np.array(actions)

    def store_transition(self, s, a, r, s2, d):
        self.buffer.push((s, a, r, s2, d))

    def update(self):
        # Only learn after warm-up and when enough samples
        if self.step_count < self.learning_starts or len(self.buffer) < self.batch_size:
            return

        # Sample a batch
        (states, actions, rewards, next_states, dones), _, _ = \
            self.buffer.sample(self.batch_size)

        # Move to device
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device).squeeze(1)  # [B, ndims]
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        qvals = self.qnet(states)  # [B, sum(nvec)]
        splits = torch.split(qvals, self.nvec, dim=1)
        q_taken = [
            chunk.gather(1, actions[:, i].unsqueeze(1))
            for i, chunk in enumerate(splits)
        ]
        current_q = torch.cat(q_taken, dim=1).sum(dim=1)  # [B]

        # Compute target Q-values
        with torch.no_grad():
            next_qvals = self.target(next_states)
            next_splits = torch.split(next_qvals, self.nvec, dim=1)
            max_next = [chunk.max(dim=1)[0] for chunk in next_splits]
            max_next = torch.stack(max_next, dim=1).sum(dim=1)
            target_q = rewards + (1 - dones) * self.gamma * max_next

        # Loss and optimization
        loss = nn.MSELoss()(current_q, target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Periodic target network update
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.qnet.state_dict())

    def update_target(self):
        self.target.load_state_dict(self.qnet.state_dict())

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)
