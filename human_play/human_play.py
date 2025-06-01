import pygame
import numpy as np
from AppleGame import AppleGameEnv

def human_play(env, golden_grids, output_file, start_idx=1):
    """
    Allows humans to play a subset of golden grids with a 120-second time limit per grid and saves scores.
    Pressing the spacebar ends the current grid and shows the "Next Grid" button.
    Clicking the "Next Grid" button advances to the next grid.
    
    Args:
        env: AppleGameEnv instance with render_mode='human'
        golden_grids: List of numpy arrays representing the grids
        output_file: File path to save scores
        start_idx: Grid index for display (1-based)
    """
    scores = []
    total_grids = 100
    clock = pygame.time.Clock()
    
    # Initialize everything ASAP
    print("Initializing visualizer...")
    env.visualizer.initialize()
    print("Visualizer ready!")

    for idx, grid in enumerate(golden_grids):
        global_idx = start_idx + idx
        print(f"Starting grid {global_idx}")
        
        # Directly update game state without expensive reset
        env.game.grid = grid.copy()
        env.game.score = 0
        env.game.time_remaining = env.game.time_limit
        env.game.game_over = False
        env.feasible_actions = env.game.get_feasible_actions()
        
        env.render()
        
        selection_start = None
        selection_end = None
        last_time = pygame.time.get_ticks() / 1000.0
        state = "playing"
        advance_to_next_grid = False  # Flag to control grid advancement

        while not advance_to_next_grid:
            current_time = pygame.time.get_ticks() / 1000.0
            dt = current_time - last_time
            last_time = current_time

            if state == "playing":
                env.game.update_time(dt)
                if env.game.time_remaining <= 0:
                    env.game.time_remaining = 0
                    env.game.game_over = True
                    state = "game_over"
                    print(f"Grid {global_idx} ended due to time limit")

            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print(f"Exiting game at grid {global_idx}")
                        env.close()
                        with open(output_file, 'w') as f:
                            for i, score in enumerate(scores):
                                f.write(f"{start_idx + i},{score}\n")
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        # Spacebar ends the current grid
                        if state == "playing":
                            env.game.game_over = True
                            state = "game_over"
                            print(f"Spacebar pressed, ending grid {global_idx}")
                    elif state == "game_over" and event.type == pygame.MOUSEBUTTONDOWN:
                        # Check if click is within button bounds
                        button_rect = env.visualizer.get_button_rect()
                        if button_rect:
                            if button_rect.collidepoint(event.pos):
                                print(f"Next Grid button clicked at {event.pos}, advancing to grid {global_idx + 1}")
                                advance_to_next_grid = True  # Set flag to exit main loop
                                break  # Break out of event loop
                            else:
                                print(f"Click at {event.pos} outside button bounds {button_rect}")
                        else:
                            print("Button rectangle is None")
                    elif state == "playing":
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            selection_start = event.pos
                        elif event.type == pygame.MOUSEMOTION and selection_start:
                            selection_end = event.pos
                        elif event.type == pygame.MOUSEBUTTONUP and selection_start and selection_end:
                            cell_size = env.visualizer.cell_size
                            start_x = selection_start[0] // cell_size
                            start_y = selection_start[1] // cell_size
                            end_x = selection_end[0] // cell_size
                            end_y = selection_end[1] // cell_size
                            x1 = min(start_x, end_x)
                            y1 = min(start_y, end_y)
                            x2 = max(start_x, end_x)
                            y2 = max(start_y, end_y)
                            x1 = max(0, min(x1, env.width - 1))
                            y1 = max(0, min(y1, env.height - 1))
                            x2 = max(0, min(x2, env.width - 1))
                            y2 = max(0, min(y2, env.height - 1))
                            reward, valid = env.game.make_selection(x1, y1, x2, y2)
                            if valid:
                                env.feasible_actions = env.game.get_feasible_actions()
                                if not env.feasible_actions:
                                    env.game.game_over = True
                                    state = "game_over"
                                    print(f"Grid {global_idx} ended due to no feasible actions")
                            selection_start = None
                            selection_end = None
            except Exception as e:
                print(f"Error processing event in grid {global_idx}: {e}")
                env.close()
                with open(output_file, 'w') as f:
                    for i, score in enumerate(scores):
                        f.write(f"{start_idx + i},{score}\n")
                return

            if selection_start and selection_end:
                env.visualizer.set_selection(selection_start, selection_end)
            else:
                env.visualizer.clear_selection()

            env.visualizer.message = {
                "grid_idx": global_idx,
                "total_grids": total_grids,
                "state": state
            }
            env.render()
            clock.tick(30)

        # Save score when moving to next grid
        scores.append(env.game.score)
        print(f"Score {env.game.score} saved for grid {global_idx}")
        
        # Clear selection for next grid (no need to close visualizer)
        env.visualizer.clear_selection()
        print(f"Moving to next grid")

    env.close()
    with open(output_file, 'w') as f:
        for i, score in enumerate(scores):
            f.write(f"{start_idx + i},{score}\n")
    print(f"Saved scores to {output_file}")

if __name__ == "__main__":
    pygame.init()
    try:
        golden_grids = np.load('golden_grids.npy')
        env = AppleGameEnv(width=17, height=10, render_mode='human')
        human_play(env, golden_grids[0:33], 'human_scores_1_33.txt', start_idx=1)
        human_play(env, golden_grids[33:66], 'human_scores_34_66.txt', start_idx=34)
        human_play(env, golden_grids[66:100], 'human_scores_67_100.txt', start_idx=67)
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        env.close()
        pygame.quit()
