import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time

from agents.random_agent import RandomAgent
from agents.dqn_agent      import DQNAgent
# from agents.reinforce_agent import REINFORCEAgent

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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Apple Game')
    parser.add_argument('--width', type=int, default=17, help='Width of the grid')
    parser.add_argument('--height', type=int, default=10, help='Height of the grid')
    parser.add_argument('--mode', choices=['play', 'rl'], default='rl', help='Play mode or RL mode')
    parser.add_argument('--agent',  choices=['random', 'dqn', 'reinforce'], default='random',
                        help='Which agent to run in RL mode')
    parser.add_argument('--render_mode', choices=['human', 'agent'], default='human', help='Render the game')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--time_limit', type=int, default=120, help='Time limit for the game')
    
    args = parser.parse_args()
    
    if args.mode == 'play':
        # Play the game manually
        play_game(width=args.width, height=args.height, time_limit=args.time_limit)
    else:
        # RL mode demonstration
        # RL mode with chosen agent
        env = AppleGameEnv(width=args.width, height=args.height, render_mode=args.render_mode, time_limit=args.time_limit)

        AGENTS = {
            'random':    RandomAgent,
            'dqn':       DQNAgent,
            # 'reinforce': REINFORCEAgent,
        }

        AgentClass = AGENTS[args.agent]
        agent = AgentClass(env)
        print(f"Running RL with {args.agent} agent")

        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _, info = env.step(action)

            # if the agent supports learning, store and update
            if hasattr(agent, 'store_transition'):
                agent.store_transition(obs, action, reward, next_obs, done)
            if hasattr(agent, 'update'):
                agent.update()

            obs = next_obs
            env.render()
            print(f"Action: {action}, Reward: {reward}, Info: {info}")
            # time.sleep(0.5)

        agent.save('dqn_apple_game.pth')
        env.close()

if __name__ == "__main__":
    main()
