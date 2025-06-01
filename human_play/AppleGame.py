import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import os

class AppleGame:
    def __init__(self, width=17, height=10, time_limit=120):
        self.width = width
        self.height = height
        self.time_limit = time_limit
        self.reset()

    def reset(self, fixed_grid=None):
        if fixed_grid is not None:
            self.grid = fixed_grid.copy()
        else:
            self.grid = np.random.randint(1, 10, size=(self.height, self.width))
        self.score = 0
        self.time_remaining = self.time_limit
        self.game_over = False
        return self.grid.copy()

    def is_valid_selection(self, x1, y1, x2, y2):
        if not (0 <= x1 <= x2 < self.width and 0 <= y1 <= y2 < self.height):
            return False
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        return np.sum(rectangle) == 10

    def get_rectangle_sum(self, x1, y1, x2, y2):
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        return np.sum(rectangle)

    def make_selection(self, x1, y1, x2, y2):
        if self.game_over:
            return 0, True
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if not self.is_valid_selection(x1, y1, x2, y2):
            return 0, False
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        num_apples = (rectangle > 0).sum()
        self.grid[y1:y2+1, x1:x2+1] = 0
        self.score += num_apples
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

    def get_feasible_actions(self):
        feasible = []
        for x1 in range(self.width):
            for y1 in range(self.height):
                for x2 in range(x1, self.width):
                    for y2 in range(y1, self.height):
                        if self.get_rectangle_sum(x1, y1, x2, y2) == 10:
                            rect_width = x2 - x1 + 1
                            rect_height = y2 - y1 + 1
                            feasible.append((x1, y1, rect_width, rect_height))
        return feasible

class PyGameVisualizer:
    def __init__(self, game, cell_size=50, save_frames=False, frame_dir='/content/frames'):
        self.game = game
        self.cell_size = cell_size
        self.width = game.width * cell_size
        self.height = game.height * cell_size
        self.initialized = False
        self.selection_start = None
        self.selection_end = None
        self.save_frames = save_frames
        self.frame_dir = frame_dir
        self.message = None
        self.button_rect = None
        if save_frames and not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Apple Game")
        self.font = pygame.font.Font(None, 36)
        self.initialized = True

    def get_button_rect(self):
        return self.button_rect

    def render(self):
        if not self.initialized:
            self.initialize()
        self.screen.fill((255, 255, 255))

        # Draw grid cells
        for y in range(self.game.height):
            for x in range(self.game.width):
                value = self.game.grid[y, x]
                if value > 0:
                    pygame.draw.rect(
                        self.screen, (255, 0, 0),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    text = self.font.render(str(value), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(
                        self.screen, (200, 200, 200),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1
                    )

        # Draw selection rectangle
        if self.selection_start and self.selection_end:
            x1 = min(self.selection_start[0], self.selection_end[0])
            y1 = min(self.selection_start[1], self.selection_end[1])
            x2 = max(self.selection_start[0], self.selection_end[0])
            y2 = max(self.selection_start[1], self.selection_end[1])
            grid_x1 = x1 // self.cell_size
            grid_y1 = y1 // self.cell_size
            grid_x2 = x2 // self.cell_size
            grid_y2 = y2 // self.cell_size
            rect_sum = self.game.get_rectangle_sum(grid_x1, grid_y1, grid_x2, grid_y2)
            color = (0, 255, 0, 100) if rect_sum == 10 else (255, 0, 0, 100)
            pixel_x1 = grid_x1 * self.cell_size
            pixel_y1 = grid_y1 * self.cell_size
            pixel_x2 = (grid_x2 + 1) * self.cell_size
            pixel_y2 = (grid_y2 + 1) * self.cell_size
            s = pygame.Surface((pixel_x2 - pixel_x1, pixel_y2 - pixel_y1), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (pixel_x1, pixel_y1))

        # Draw UI elements
        score_text = self.font.render(f"Score: {self.game.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        if self.message:
            grid_idx = self.message.get("grid_idx", 1)
            total_grids = self.message.get("total_grids", 100)
            state = self.message.get("state", "playing")
            # Time and grid number (top-right)
            time_grid_text = f"Grid: {grid_idx}/{total_grids}  Time: {int(self.game.time_remaining)}s"
            time_grid_surface = self.font.render(time_grid_text, True, (0, 0, 0))
            time_grid_rect = time_grid_surface.get_rect(topright=(self.width - 10, 10))
            self.screen.blit(time_grid_surface, time_grid_rect)
            
            # Game over message and button
            if state == "game_over":
                game_over_text = "Game Over!"
                game_over_surface = self.font.render(game_over_text, True, (0, 0, 0))
                game_over_rect = game_over_surface.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(game_over_surface, game_over_rect)

                # Draw "Next Grid" button
                button_text = "Next Grid"
                button_surface = self.font.render(button_text, True, (255, 255, 255))
                button_padding = 10
                button_rect = button_surface.get_rect(
                    center=(self.width // 2, self.height // 2 + 50)
                )
                button_bg_rect = button_rect.inflate(button_padding, button_padding)
                pygame.draw.rect(self.screen, (0, 128, 0), button_bg_rect)  # Green button
                self.screen.blit(button_surface, button_rect)
                self.button_rect = button_bg_rect
            else:
                self.button_rect = None

        if self.save_frames:
            pygame.image.save(self.screen, os.path.join(self.frame_dir, f'frame_{id(self.game):04d}.png'))

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

    def __init__(self, width=17, height=10, render_mode=None, max_actions=1000, save_frames=False):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.max_actions = max_actions
        self.game = AppleGame(width=width, height=height)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=9, shape=(height, width), dtype=np.int32)
        self.feasible_actions = []
        self.visualizer = None
        if render_mode == 'human':
            self.visualizer = PyGameVisualizer(self.game, save_frames=save_frames)

    def reset(self, seed=None, options=None, fixed_grid=None):
        observation = self.game.reset(fixed_grid=fixed_grid)
        self.feasible_actions = self.game.get_feasible_actions()
        self.action_space = spaces.Discrete(len(self.feasible_actions) if self.feasible_actions else 1)
        if self.visualizer and self.visualizer.initialized:
            self.visualizer.close()
            self.visualizer = PyGameVisualizer(self.game, save_frames=self.visualizer.save_frames)
        return observation, {'feasible_actions': self.feasible_actions}

    def step(self, action_idx):
        self.action_space = spaces.Discrete(len(self.feasible_actions) if self.feasible_actions else 1)
        if not self.feasible_actions:
            reward = -0.1
            observation = self.game.grid.copy()
            done = True
            info = {
                'score': self.game.score,
                'time_remaining': self.game.time_remaining,
                'valid_selection': False,
                'feasible_actions': self.feasible_actions
            }
            return observation, reward, done, False, info
        if action_idx >= len(self.feasible_actions):
            action_idx = 0
        x1, y1, rect_width, rect_height = self.feasible_actions[action_idx]
        x2 = min(x1 + rect_width - 1, self.width - 1)
        y2 = min(y1 + rect_height - 1, self.height - 1)
        reward, valid = self.game.make_selection(x1, y1, x2, y2)
        if valid:
            self.game.update_time(1)
        reward -= 0.01
        observation = self.game.grid.copy()
        done = self.game.game_over
        self.feasible_actions = self.game.get_feasible_actions()
        self.action_space = spaces.Discrete(len(self.feasible_actions) if self.feasible_actions else 1)
        info = {
            'score': self.game.score,
            'time_remaining': self.game.time_remaining,
            'valid_selection': valid,
            'feasible_actions': self.feasible_actions
        }
        return observation, reward, done, False, info

    def render(self):
        if self.visualizer:
            self.visualizer.render()
            if self.render_mode == 'rgb_array':
                return pygame.surfarray.array3d(self.visualizer.screen)

    def close(self):
        if self.visualizer:
            self.visualizer.close()

if __name__ == "__main__":
    width = 17
    height = 10
    num_grids = 100
    golden_grids = []
    for _ in range(num_grids):
        game = AppleGame(width=width, height=height)
        grid = game.reset()
        golden_grids.append(grid)
    np.save('golden_grids.npy', np.array(golden_grids))
    print(f"Saved {num_grids} golden grids to 'golden_grids.npy'")