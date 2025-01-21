import pygame
import numpy as np
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import time

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)  # I piece
YELLOW = (255, 255, 0)  # O piece
PURPLE = (128, 0, 128)  # T piece
BLUE = (0, 0, 255)  # J piece
ORANGE = (255, 165, 0)  # L piece
GREEN = (0, 255, 0)  # S piece
RED = (255, 0, 0)  # Z piece

# Block size
BLOCK_SIZE = 30
PADDING = 1

class TetrisVisualizer:
    def __init__(self, env):
        self.env = env
        pygame.init()
        
        # Calculate window dimensions
        self.width = env.BOARD_WIDTH * BLOCK_SIZE
        self.height = env.BOARD_HEIGHT * BLOCK_SIZE
        
        # Create the game window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Tetris AI')
        
        # Color mapping for different pieces
        self.colors = {
            0: BLACK,  # Empty space
            1: WHITE,  # Active piece
            2: CYAN,   # Frozen piece
        }

    def draw_board(self, state):
        self.screen.fill(BLACK)
        
        # Draw the grid
        for y in range(self.env.BOARD_HEIGHT):
            for x in range(self.env.BOARD_WIDTH):
                value = state[y][x]
                color = self.colors[min(2, value)]  # Use 2 for frozen pieces
                
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE - PADDING, BLOCK_SIZE - PADDING)
                )
        
        pygame.display.flip()

def watch_ai_play():
    env = TetrisEnv()
    state_size = env.BOARD_HEIGHT * env.BOARD_WIDTH
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    
    # Load trained model weights here if you have them
    # agent.model.load_state_dict(torch.load('tetris_model.pth'))
    
    visualizer = TetrisVisualizer(env)
    
    while True:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get AI action
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Update visualization
            visualizer.draw_board(next_state)
            total_reward += reward
            state = next_state
            
            # Control game speed
            time.sleep(0.1)
        
        print(f"Game Over! Score: {total_reward}")
        time.sleep(1)

def play_human():
    env = TetrisEnv()
    visualizer = TetrisVisualizer(env)
    
    while True:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_UP:
                        action = 2
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    
                    if action is not None:
                        next_state, reward, done = env.step(action)
                        total_reward += reward
                        state = next_state
            
            # Update visualization
            visualizer.draw_board(state)
            time.sleep(0.05)
        
        print(f"Game Over! Score: {total_reward}")
        time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ai', 'human'], default='human',
                       help='Play mode: ai or human')
    args = parser.parse_args()
    
    if args.mode == 'ai':
        watch_ai_play()
    else:
        play_human() 