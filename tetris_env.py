import numpy as np
import random

class TetrisEnv:
    # Tetris board dimensions
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    
    # Define Tetris pieces (tetriminoes)
    PIECES = [
        [[1, 1, 1, 1]],  # _
        [[1, 1], [1, 1]],  # O
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1, 0], [0, 1, 1]],  # S
        [[0, 1, 1], [1, 1, 0]],   # Z
        [[0,1,0], [1,1,1]]  # T
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH))
        self.current_piece = self._get_random_piece()
        self.current_pos = [0, self.BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.game_over = False
        self.score = 0
        return self._get_state()

    def _get_random_piece(self):
        return np.array(random.choice(self.PIECES))

    def _get_state(self):
        # Create a copy of the board for the current state
        state = self.board.copy()
        
        # Add current piece to the state
        piece_height, piece_width = self.current_piece.shape
        y, x = self.current_pos
        
        # Only add the piece if it's in bounds
        if (0 <= y < self.BOARD_HEIGHT - piece_height + 1 and 
            0 <= x < self.BOARD_WIDTH - piece_width + 1):
            state[y:y+piece_height, x:x+piece_width] += self.current_piece
            
        return state

    def step(self, action):
        """
        Actions:
        0: Move left
        1: Move right
        2: Rotate
        3: Move down
        4: Push to bottom
        """
        reward = 0
        done = False

        if action == 0:  # Move left
            self._move(-1)
        elif action == 1:  # Move right
            self._move(1)
        elif action == 2:  # Rotate
            self._rotate()
        elif action == 3:  # Move down
            if not self._move_down():
                self._freeze_piece()
                cleared_lines = self._clear_lines()
                reward = self._calculate_reward(cleared_lines)
                self.current_piece = self._get_random_piece()
                self.current_pos = [0, self.BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
                
                if self._check_collision():
                    done = True
                    reward = -10  # Penalty for game over

        elif action == 4:  # Push to bottom
            self._push_to_bottom()
            self._freeze_piece()
            cleared_lines = self._clear_lines()
            reward = self._calculate_reward(cleared_lines)
            self.current_piece = self._get_random_piece()
            self.current_pos = [0, self.BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]

        if self._check_collision():
            done = True
            reward = -10  # Penalty for game over

        return self._get_state(), reward, done

    def _move(self, dx):
        x, y = self.current_pos[1], self.current_pos[0]
        new_x = x + dx
        if self._is_valid_move(y, new_x):
            self.current_pos[1] = new_x
            return True
        return False

    def _rotate(self):
        rotated_piece = np.rot90(self.current_piece)
        if self._is_valid_move(self.current_pos[0], self.current_pos[1], rotated_piece):
            self.current_piece = rotated_piece

    def _move_down(self):
        y = self.current_pos[0]
        new_y = y + 1
        if self._is_valid_move(new_y, self.current_pos[1]):
            self.current_pos[0] = new_y
            return True
        return False
    
    def _push_to_bottom(self):
        while self._move_down():
            pass

    def _is_valid_move(self, y, x, piece=None):
        if piece is None:
            piece = self.current_piece
        piece_height, piece_width = piece.shape
        
        # Check boundaries
        if (x < 0 or x + piece_width > self.BOARD_WIDTH or 
            y < 0 or y + piece_height > self.BOARD_HEIGHT):
            return False

        # Check collision with other pieces
        for i in range(piece_height):
            for j in range(piece_width):
                if piece[i][j] and self.board[y + i][x + j]:
                    return False
        return True

    def _freeze_piece(self):
        piece_height, piece_width = self.current_piece.shape
        y, x = self.current_pos
        self.board[y:y+piece_height, x:x+piece_width] += self.current_piece

    def _clear_lines(self):
        lines_cleared = 0
        y = self.BOARD_HEIGHT - 1
        while y >= 0:
            if np.all(self.board[y]):
                self.board = np.vstack((np.zeros(self.BOARD_WIDTH), self.board[:y], self.board[y+1:]))
                lines_cleared += 1
            else:
                y -= 1
        return lines_cleared

    def _calculate_reward(self, lines_cleared):
        if lines_cleared == 0:
            return 1  # Small reward for placing a piece
        return lines_cleared * 100  # Larger reward for clearing lines

    def _check_collision(self):
        return not self._is_valid_move(self.current_pos[0], self.current_pos[1]) 