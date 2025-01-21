from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import numpy as np
import torch
import os

def train():
    env = TetrisEnv()
    state_size = env.BOARD_HEIGHT * env.BOARD_WIDTH
    action_size = 4  # left, right, rotate, down
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000
    target_update_frequency = 10

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break

        if episode % target_update_frequency == 0:
            agent.update_target_model()

        print(f"Episode: {episode + 1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    model_path = 'tetris_model.pth'
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train() 