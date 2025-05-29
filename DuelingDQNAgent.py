import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time

class AppleGame:
    def __init__(self, width=17, height=10, time_limit=120):
        self.width = width
        self.height = height
        self.time_limit = time_limit
        self.reset()

    def reset(self):
        # Initialize grid with random numbers from 1-9
        self.grid = np.random.randint(1, 10, size=(self.height, self.width))
        self.score = 0
        self.time_remaining = self.time_limit
        self.game_over = False
        return self.grid.copy()

    def is_valid_selection(self, x1, y1, x2, y2):
        # Check if coordinates are valid
        if not (0 <= x1 <= x2 < self.width and 0 <= y1 <= y2 < self.height):
            return False

        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]

        # Check if sum equals 10
        return np.sum(rectangle) == 10

    def get_rectangle_sum(self, x1, y1, x2, y2):
        # Ensure coordinates are valid
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))

        # Ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]

        # Return sum
        return np.sum(rectangle)

    def make_selection(self, x1, y1, x2, y2):
        if self.game_over:
            return 0, True

        # Ensure coordinates are valid
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))

        # Ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if not self.is_valid_selection(x1, y1, x2, y2):
            return 0, False

        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]

        # Count apples in the rectangle
        num_apples = (rectangle > 0).sum()

        # Clear the apples (set to 0)
        self.grid[y1:y2+1, x1:x2+1] = 0

        # Update score
        self.score += num_apples

        # Check if all apples are cleared
        if (self.grid == 0).all():
            self.game_over = True

        return num_apples, True

    def update_time(self, dt):
        if self.game_over:
            return

        self.time_remaining -= dt
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True


class PyGameVisualizer:
    def __init__(self, game, cell_size=50):
        self.game = game
        self.cell_size = cell_size
        self.width = game.width * cell_size
        self.height = game.height * cell_size
        self.initialized = False
        self.selection_start = None
        self.selection_end = None

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Apple Game")
        self.font = pygame.font.Font(None, 36)
        self.initialized = True

    def render(self):
        if not self.initialized:
            self.initialize()

        self.screen.fill((255, 255, 255))

        # Draw grid
        for y in range(self.game.height):
            for x in range(self.game.width):
                value = self.game.grid[y, x]
                if value > 0:
                    # Draw apple
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    # Draw number
                    text = self.font.render(str(value), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
                else:
                    # Draw empty cell
                    pygame.draw.rect(
                        self.screen,
                        (200, 200, 200),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1
                    )

        # Draw current selection
        if self.selection_start and self.selection_end:
            x1 = min(self.selection_start[0], self.selection_end[0])
            y1 = min(self.selection_start[1], self.selection_end[1])
            x2 = max(self.selection_start[0], self.selection_end[0])
            y2 = max(self.selection_start[1], self.selection_end[1])

            # Convert to grid coordinates
            grid_x1 = x1 // self.cell_size
            grid_y1 = y1 // self.cell_size
            grid_x2 = x2 // self.cell_size
            grid_y2 = y2 // self.cell_size

            # Calculate sum
            rect_sum = self.game.get_rectangle_sum(grid_x1, grid_y1, grid_x2, grid_y2)

            # Draw rectangle
            color = (0, 255, 0, 128) if rect_sum == 10 else (255, 0, 0, 128)

            # Convert back to pixel coordinates for drawing
            pixel_x1 = grid_x1 * self.cell_size
            pixel_y1 = grid_y1 * self.cell_size
            pixel_x2 = (grid_x2 + 1) * self.cell_size
            pixel_y2 = (grid_y2 + 1) * self.cell_size

            # Create a surface with alpha channel
            s = pygame.Surface((pixel_x2 - pixel_x1, pixel_y2 - pixel_y1), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (pixel_x1, pixel_y1))

            # Draw sum
            sum_text = self.font.render(f"Sum: {rect_sum}", True, (0, 0, 0))
            self.screen.blit(sum_text, (10, self.height - 40))

        # Draw score
        score_text = self.font.render(f"Score: {self.game.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        # Draw time remaining
        time_text = self.font.render(f"Time: {int(self.game.time_remaining)}", True, (0, 0, 0))
        self.screen.blit(time_text, (self.width - 150, 10))

        pygame.display.flip()

    def set_selection(self, start_pos, end_pos):
        self.selection_start = start_pos
        self.selection_end = end_pos

    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None

    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False


class AppleGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, width=17, height=10, render_mode=None, time_limit=120):
        super().__init__()

        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.time_limit = time_limit

        # Create the game
        self.game = AppleGame(width=width, height=height, time_limit=time_limit)

        # Define action space (top-left x, top-left y, rectangle width, rectangle height)
        self.action_space = spaces.MultiDiscrete([width, height, width, height])

        # Define observation space (grid of apples)
        self.observation_space = spaces.Box(low=0, high=9, shape=(height, width), dtype=np.int32)

        # Create visualizer if needed
        self.visualizer = None
        if render_mode == 'human':
            self.visualizer = PyGameVisualizer(self.game)
            self.visualizer.initialize()

    def reset(self, seed=None, options=None):
        # Reset the game
        observation = self.game.reset()

        # Reset the renderer
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = PyGameVisualizer(self.game)

        return observation, {}

    def step(self, action):
        # Parse action
        x1, y1, rect_width, rect_height = action

        # Ensure width and height are at least 1
        rect_width = max(1, rect_width)
        rect_height = max(1, rect_height)

        # Calculate bottom-right coordinates
        x2 = min(x1 + rect_width - 1, self.width - 1)
        y2 = min(y1 + rect_height - 1, self.height - 1)

        # Make selection
        reward, valid = self.game.make_selection(x1, y1, x2, y2)

        # Update time (assume 1 second per step)
        # if valid:
        #     self.game.update_time(1)
        self.game.update_time(0.1)

        # Adjust reward for RL
        if not valid:
            reward = -0.1  # Penalty for invalid selection

        # Small penalty for each step to encourage faster solving
        reward -= 0.01

        # Get observation
        observation = self.game.grid.copy()

        # Check if game is over
        done = self.game.game_over

        # Additional info
        info = {
            'score': self.game.score,
            'time_remaining': self.game.time_remaining,
            'valid_selection': valid
        }

        return observation, reward, done, False, info

    def render(self):
        if self.visualizer:
            self.visualizer.render()

    def close(self):
        if self.visualizer:
            font = pygame.font.Font(None, 72)
            text = font.render("Game Over!", True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.visualizer.width // 2, self.visualizer.height // 2))
            self.visualizer.screen.blit(text, text_rect)

            # Show final score
            score_font = pygame.font.Font(None, 48)
            score_text = score_font.render(f"Final Score: {self.game.score}", True, (0, 0, 0))
            score_rect = score_text.get_rect(center=(self.visualizer.width // 2, self.visualizer.height // 2 + 50))
            self.visualizer.screen.blit(score_text, score_rect)

            pygame.display.flip()

            # Wait for user to close the window
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
            self.visualizer.close()

def play_game(width=17, height=10, time_limit=120):
    # Create the game
    game = AppleGame(width=width, height=height, time_limit=time_limit)
    visualizer = PyGameVisualizer(game)
    visualizer.initialize()

    # Variables for rectangle selection
    selecting = False

    clock = pygame.time.Clock()
    running = True
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start selection
                selecting = True
                visualizer.set_selection(event.pos, event.pos)
            elif event.type == pygame.MOUSEMOTION and selecting:
                # Update selection
                visualizer.set_selection(visualizer.selection_start, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and selecting:
                # End selection
                selecting = False

                # Convert to grid coordinates
                x1 = visualizer.selection_start[0] // visualizer.cell_size
                y1 = visualizer.selection_start[1] // visualizer.cell_size
                x2 = event.pos[0] // visualizer.cell_size
                y2 = event.pos[1] // visualizer.cell_size

                # Ensure x1 <= x2 and y1 <= y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Make selection
                game.make_selection(x1, y1, x2, y2)

                # Clear selection
                visualizer.clear_selection()

        # Update time (assume 0.1 seconds per frame)
        game.update_time(0.1)

        # Render
        visualizer.render()

        # Cap at 10 FPS
        clock.tick(10)

    # Game over
    if game.game_over:
        font = pygame.font.Font(None, 72)
        text = font.render("Game Over!", True, (255, 0, 0))
        text_rect = text.get_rect(center=(visualizer.width // 2, visualizer.height // 2))
        visualizer.screen.blit(text, text_rect)

        # Show final score
        score_font = pygame.font.Font(None, 48)
        score_text = score_font.render(f"Final Score: {game.score}", True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(visualizer.width // 2, visualizer.height // 2 + 50))
        visualizer.screen.blit(score_text, score_rect)

        pygame.display.flip()

        # Wait for user to close the window
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False

    visualizer.close()


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
        # transition = (state, flat_action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device='cpu'):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.stack(states)).to(device)
        actions     = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards     = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(device)
        dones       = torch.FloatTensor(np.array(dones)).to(device)

        return (states, actions, rewards, next_states, dones)


class DuelingConvQNet(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super().__init__()
        h, w = input_shape
        # Shared convolutional feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute flattened feature size
        dummy = torch.zeros(1, 1, h, w)
        feat_size = self.feature(dummy).shape[1]
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(feat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [B, H, W]
        x = x.unsqueeze(1).float()  # [B,1,H,W]
        feats = self.feature(x)     # [B, feat_size]
        value = self.value_stream(feats)                # [B,1]
        advantage = self.adv_stream(feats)              # [B, n_actions]
        qvals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return qvals  # [B, n_actions]


class DQNAgent:
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
        self.nvec = tuple(env.action_space.nvec.tolist())
        self.n_actions = int(np.prod(self.nvec))

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        # Dueling Q-network & target
        self.qnet = DuelingConvQNet(obs_shape, hidden_dim, self.n_actions).to(device)
        self.target = DuelingConvQNet(obs_shape, hidden_dim, self.n_actions).to(device)
        self.target.load_state_dict(self.qnet.state_dict())

        self.opt = optim.AdamW(self.qnet.parameters(), lr=lr)
        self.gamma = gamma

        # ε-greedy
        self.epsilon     = epsilon_start
        self.eps_start   = epsilon_start
        self.eps_end     = epsilon_end
        self.eps_decay   = epsilon_decay

        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.device     = device
        self.env        = env

    def get_action(self, state, is_training=True):
        self.step_count += 1
        if is_training and self.step_count > self.learning_starts:
            self.epsilon = max(
                self.eps_end,
                self.eps_start - (self.step_count - self.learning_starts)
                * (self.eps_start - self.eps_end) / self.eps_decay
            )
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qvals = self.qnet(x).squeeze(0)
            idx   = qvals.argmax().item()
        action = np.unravel_index(idx, self.nvec)
        return np.array(action, dtype=int)

    def store_transition(self, s, a, r, s2, done):
        # multi-dim action a -> flat index
        flat_a = np.ravel_multi_index(tuple(a), dims=self.nvec)
        self.buffer.push((s, flat_a, r, s2, float(done)))

    def update(self):
        if self.step_count < self.learning_starts or len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size, self.device)

        # 현재 Q(s,a)
        qvals      = self.qnet(states)
        current_q  = qvals.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_qvals = self.target(next_states)
            max_next   = next_qvals.max(dim=1)[0]
            target_q   = rewards + (1 - dones) * self.gamma * max_next

        loss = nn.MSELoss()(current_q, target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.qnet.state_dict())

import os
import argparse
import numpy as np
import torch
import logging
from tensorboard import notebook
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Apple Game DQN Train/Eval with Early Stopping')
    # Environment parameters
    parser.add_argument('--width',         type=int,   default=17,     help='Grid width')
    parser.add_argument('--height',        type=int,   default=10,     help='Grid height')
    parser.add_argument('--render_mode',   choices=['human','agent'], default='human', help='Render mode')
    parser.add_argument('--time_limit',    type=int,   default=120,    help='Time limit per episode (s)')
    parser.add_argument('--seed',          type=int,   default=None,   help='Random seed')

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='Run training')
    mode.add_argument('--eval',  action='store_true', help='Run evaluation')

    # Training control
    parser.add_argument('--episodes',    type=int,   default=1000,   help='Maximum number of episodes')
    parser.add_argument('--patience',    type=int,   default=20,     help='Early-stop patience (episodes)')
    parser.add_argument('--min_delta',   type=float, default=1e-3,   help='Minimum improvement to reset patience')
    parser.add_argument('--learning_starts',     type=int,   default=50000, help='Replay warm-up frames')
    parser.add_argument('--eps_decay',           type=int,   default=10000, help='Epsilon decay frames')
    parser.add_argument('--target_update_steps', type=int,   default=500,   help='Target sync freq')
    parser.add_argument('--save_freq',           type=int,   default=100,   help='Save checkpoint every N episodes')
    parser.add_argument('--ckpt_path',           type=str,   default=None,  help='Resume checkpoint path')

    # Evaluation control
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of evaluation episodes')

    # Agent hyperparameters
    parser.add_argument('--lr',            type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim',    type=int,   default=256,   help='Hidden layer size')
    parser.add_argument('--gamma',         type=float, default=0.99,  help='Discount factor')
    parser.add_argument('--buffer_size',   type=int,   default=50000, help='Replay buffer capacity')
    parser.add_argument('--batch_size',    type=int,   default=32,    help='Batch size')
    parser.add_argument('--epsilon_start', type=float, default=1.0,   help='Initial epsilon')
    parser.add_argument('--epsilon_end',   type=float, default=0.01,  help='Final epsilon')
    parser.add_argument('--device',        type=str,   default='cpu',  help='Torch device')
    return parser.parse_args()

def setup_logging(run_id: str):
    log_dir = os.path.join("logs", f"{run_id}_DQN")
    os.makedirs(log_dir, exist_ok=True)
    # File-only logger
    logger = logging.getLogger("AppleGameDQN")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    return logger, tb_writer, log_dir

def evaluate(agent, env, episodes, seed=None):
    scores = []
    pbar = tqdm(range(1, episodes + 1), desc="Eval Episodes")
    for ep in pbar:
        obs, _ = env.reset(seed=seed)
        done, total_reward = False, 0.0
        while not done:
            action = agent.get_action(obs, is_training=False)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            env.render()
        final_score = info.get("score", total_reward)
        scores.append(final_score)
        pbar.set_postfix({"score": f"{final_score:.2f}", "avg": f"{np.mean(scores):.2f}", "std": f"{np.std(scores):.2f}"})
    avg, std = np.mean(scores), np.std(scores)
    print(f"\nEvaluation over {episodes} episodes → avg: {avg:.2f}, std: {std:.2f}")
    return scores

def main(args):
    # args = parse_args()
    # seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, tb_writer, log_dir = setup_logging(run_id)

    tb_dir = f'{log_dir}/tensorboard'
    port = 6006
    notebook.start(f'--logdir {tb_dir} --host 0.0.0.0')

    mode = "TRAIN" if args.train else "EVAL"
    logger.info(f"Mode: {mode} | Config: {args}")
    print(f"Running in {mode} mode")

    # create env & agent
    env = AppleGameEnv(
        width=args.width, height=args.height,
        render_mode=args.render_mode, time_limit=args.time_limit
    )
    agent = DQNAgent(
        env,
        lr=args.lr, hidden_dim=args.hidden_dim, gamma=args.gamma,
        buffer_size=args.buffer_size, learning_starts=(args.learning_starts if args.train else 0),
        batch_size=args.batch_size,
        epsilon_start=(args.epsilon_start if args.train else 0.0),
        epsilon_end=(args.epsilon_end if args.train else 0.0),
        epsilon_decay=(args.eps_decay if args.train else 1),
        target_update_freq=(args.target_update_steps if args.train else 0),
        device=args.device
    )
    if args.ckpt_path:
        agent.load(args.ckpt_path)
        logger.info(f"Loaded checkpoint from {args.ckpt_path}")

    if args.train:
        best_avg = -np.inf
        no_improve = 0
        scores = []

        print("Storing transitions to replay buffer...")
        tmp_steps = 0
        while tmp_steps < args.learning_starts:
            obs, _ = env.reset(seed=None)
            done = False
            while (not done) and (tmp_steps < args.learning_starts):
                action = agent.get_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                tmp_steps += 1
                env.render()

        pbar = tqdm(range(1, args.episodes + 1), desc="Train Episodes", unit="ep")
        steps = 0
        for ep in pbar:
            obs, _ = env.reset(seed=None)
            done = False

            max_steps = 1000
            cur_step = 0

            while (not done) and (cur_step < max_steps):
                action = agent.get_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                # store & update
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.update()
                steps += 1
                cur_step += 1
                obs = next_obs
                env.render()

            scores.append(info.get("score"))
            # logger.info(f"[TRAIN] Episode {ep:03d}: score = {info.get('score', 0):.2f}")
            tb_writer.add_scalar("episode_score", info.get('score', 0), ep)
            tb_writer.add_scalar("epsilon", agent.epsilon, ep)

            # early stopping check
            recent_avg = np.mean(scores[-args.patience:])
            if recent_avg > best_avg + args.min_delta:
                best_avg = recent_avg
                no_improve = 0
            else:
                no_improve += 1

            # update pbar postfix with best_avg & epsilon
            pbar.set_postfix(
                {
                    "best_avg": f"{best_avg:.2f}",
                    "score": f"{info.get('score', 0):.2f}",
                    "avg": f"{np.mean(scores[-args.patience:]):.2f}",
                    "std": f"{np.std(scores[-args.patience:]):.2f}",
                    "eps": f"{agent.epsilon:.2f}",
                }
            )

            # if no_improve >= args.patience:
            #     logger.info(f"Early stopping at episode {ep}: no improvement for {args.patience} eps")
            #     break

            # checkpoint
            if ep % args.save_freq == 0:
                ckpt = os.path.join(log_dir, f"dqn_ep{ep:04d}.pth")
                agent.save(ckpt)
                logger.info(f"Saved checkpoint at episode {ep}")

        # final save
        final_ckpt = os.path.join(log_dir, "dqn_final.pth")
        agent.save(final_ckpt)
        logger.info(f"Training complete at episode {ep}, best_avg = {best_avg:.2f}, model → {final_ckpt}")

    else:
        # evaluation
        agent.epsilon = 0.05
        logger.info(f'Agent epsilon: {agent.epsilon}')
        evaluate(agent, env, args.eval_episodes, seed=args.seed)

    tb_writer.close()
    env.close()


if __name__ == "__main__":
    main(parse_args())