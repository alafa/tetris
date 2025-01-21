# Tetris AI with Deep Q-Learning

A Tetris implementation that features both AI and human play modes. The AI agent is trained using Deep Q-Learning (DQN) to learn optimal Tetris strategies.

## Features

- Complete Tetris game implementation
- Deep Q-Learning AI agent with experience replay
- Real-time visualization using Pygame
- Dual play modes: AI and human
- Model saving and loading capabilities
- Configurable game parameters

## Project Structure

- `tetris_env.py`: Core Tetris game environment with game mechanics and rules
- `dqn_agent.py`: Deep Q-Network agent implementation with PyTorch
- `train.py`: Training script for the AI agent
- `visualize.py`: Game visualization and play interface
- `requirements.txt`: Project dependencies

## Requirements

- Python 3.7 or higher
- Dependencies:
  - PyTorch >= 1.9.0
  - Pygame >= 2.0.1
  - NumPy >= 1.19.0

## Installation

1. Clone the repository:
2. Create and activate a virtual environment (recommended):

#### On Windows:
```
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Training the AI

To train the AI agent from scratch:

```
python train.py
```

The training process will:
- Run for 1000 episodes by default
- Print episode scores and exploration rate
- Save the trained model as 'tetris_model.pth'
- Display training progress in the console

### Playing the Game

#### Human Mode

To play Tetris yourself:

```
python visualize.py --mode human
```

Controls:
- ←: Move piece left
- →: Move piece right
- ↑: Rotate piece
- ↓: Move piece down

#### AI Mode

To watch the trained AI play:

```
python visualize.py --mode ai
```

Note: Make sure you have trained the model first or have a pre-trained model file (`tetris_model.pth`) in the project directory.

## Game Rules

- Clear lines by filling them completely with blocks
- Score points for:
  - Placing pieces (1 point)
  - Clearing lines (100 points per line)
- Game ends when new pieces can't be placed

## Configuration

You can modify various parameters in the source files:

### tetris_env.py
- `BOARD_WIDTH`: Width of the Tetris board (default: 10)
- `BOARD_HEIGHT`: Height of the board (default: 20)

### visualize.py
- `BLOCK_SIZE`: Size of each Tetris block (default: 30)
- `PADDING`: Spacing between blocks (default: 1)

### dqn_agent.py
- `memory`: Maximum size of replay buffer (default: 10000)
- `gamma`: Discount factor (default: 0.95)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_min`: Minimum exploration rate (default: 0.01)
- `epsilon_decay`: Exploration decay rate (default: 0.995)
- `learning_rate`: Learning rate (default: 0.001)

### train.py
- `batch_size`: Training batch size (default: 32)
- `episodes`: Number of training episodes (default: 1000)
- `target_update_frequency`: How often to update target network (default: 10)

## Troubleshooting

Common issues and solutions:

1. **pygame.error: No available video device**
   - Make sure you have a display server running
   - Try running in a desktop environment

2. **CUDA out of memory**
   - Reduce batch size in train.py
   - Use CPU mode by removing CUDA device selection

3. **ModuleNotFoundError**
   - Ensure all dependencies are installed
   - Check virtual environment activation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Deep Q-Learning implementation inspired by DeepMind's DQN paper
- Tetris game mechanics based on the original 1984 game by Alexey Pajitnov