import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium import Env


class GridEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.grid_size = 100
        self.cell_size = 10
        self.screen = pygame.display.set_mode((1000, 1000))
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(4,), dtype=np.float32
        )
        self.agent_position = np.array([0, 0])
        self.reward_position = np.array([0, 0])
        self.time_limit = 10
        self.rows = 10
        self.columns = 10
        self.window_caption = "Grid Environment"
        self.fps = 60

        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption(self.window_caption)
        self.clock = pygame.time.Clock()
        self.state = None

    def _get_observation(self):
        # Convert the Pygame surface to a 2D numpy array
        array_2d = pygame.surfarray.array2d(self.screen)

        # Resize the 2D numpy array to match the observation space
        resized_array_2d = cv2.resize(
            array_2d,
            dsize=(self.observation_space.shape[1], self.observation_space.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # Reshape the 2D numpy array to match the observation space
        observation = np.reshape(resized_array_2d, self.observation_space.shape)

        return observation

    def step(self, action):
        if action == 0:
            self.player_pos[1] = max(self.player_pos[1] - 1, 0)
        elif action == 1:
            self.player_pos[1] = min(self.player_pos[1] + 1, self.columns - 1)
        elif action == 2:
            self.player_pos[0] = min(self.player_pos[0] + 1, self.rows - 1)
        elif action == 3:
            self.player_pos[0] = max(self.player_pos[0] - 1, 0)

        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_terminal()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_reward(self, state, action):
        """Return the reward for taking the given action in the given state."""
        row, col = state
        if action == 0:  # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        # Check if the new position is within the grid boundaries
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return -1  # Return negative reward for moving out of bounds

        # Check if the new position is the goal
        if (row, col) == self.goal:
            return 1  # Return positive reward for reaching the goal

        return 0  # Return zero reward for all other actions

    def reset(self):
        self.player_pos = [0, 0]
        self.state = self._get_observation()
        return self.state

    def render(self, mode="human"):
        cell_size = self.screen.get_width() // self.columns

        for row in range(self.rows):
            for col in range(self.columns):
                rect = pygame.Rect(
                    col * cell_size, row * cell_size, cell_size, cell_size
                )

                if [row, col] == self.player_pos:
                    pygame.draw.rect(self.screen, pygame.color.THECOLORS["red"], rect)
                else:
                    pygame.draw.rect(self.screen, pygame.color.THECOLORS["white"], rect)

                pygame.draw.rect(self.screen, pygame.color.THECOLORS["black"], rect, 1)

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)


gym.envs.registration.register("GridEnv-v0", entry_point="grid_env:GridEnv")
