import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    """
    A simple DQN skeleton.
    """
    def __init__(self, env,
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=10000,
                 batch_size=32,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=5000):
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n if hasattr(env.action_space, 'n') else int(np.prod(env.action_space.nvec))
        # simple MLP
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(obs_shape)), 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.target = nn.Sequential(*[l for l in self.model])
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = []
        self.buf_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.step_count = 0
        self.action_space = env.action_space

    def get_action(self, state):
        self.step_count += 1
        # epsilon-greedy
        self.epsilon = max(self.eps_end,
                           self.epsilon - (1.0 - self.eps_end) / self.eps_decay)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        with torch.no_grad():
            q = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax().item())

    def store_transition(self, s, a, r, s2, d):
        if len(self.buffer) >= self.buf_size:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = zip(*np.random.choice(self.buffer, self.batch_size))
        states, actions, rewards, next_states, dones = [torch.tensor(x) for x in batch]
        q_vals = self.model(states.float()).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states.float()).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones.float())
        loss = nn.MSELoss()(q_vals, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
