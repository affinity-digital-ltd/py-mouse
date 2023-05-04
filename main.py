import gymnasium as gym
import pygame
import numpy as np
import random
import time
from gymnasium import spaces
import os
import pickle
import itertools

pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 36)


# Custom environment
class GridEnvironment(gym.Env):
    def __init__(self):
        self.vision_range = 10
        self.time_limit = 10
        self.grid_size = 50
        self.cell_size = 18
        self.fps = 60
        self.screen = pygame.display.set_mode((1000, 1000))
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=900, shape=(4,), dtype=np.float32
        )
        self.agent_position = np.array([0, 0])
        self.reward_position = np.array([0, 0])

        # Load the chese image
        cheese_image = pygame.image.load("images/cheese.png")
        self.cheese_image = pygame.transform.scale(
            cheese_image, (self.cell_size, self.cell_size)
        )

        self.mouse_images = {
            "up": pygame.transform.scale(
                pygame.image.load("images/mouse_up.png"),
                (self.cell_size, self.cell_size),
            ),
            "down": pygame.transform.scale(
                pygame.image.load("images/mouse_down.png"),
                (self.cell_size, self.cell_size),
            ),
            "left": pygame.transform.scale(
                pygame.image.load("images/mouse_left.png"),
                (self.cell_size, self.cell_size),
            ),
            "right": pygame.transform.scale(
                pygame.image.load("images/mouse_right.png"),
                (self.cell_size, self.cell_size),
            ),
        }
        self.current_mouse_image = self.mouse_images["up"]

    def step(self, action):
        if action == 0:  # Up
            self.agent_position[1] -= self.cell_size
            self.current_mouse_image = self.mouse_images["up"]
        elif action == 1:  # Down
            self.agent_position[1] += self.cell_size
            self.current_mouse_image = self.mouse_images["down"]
        elif action == 2:  # Left
            self.agent_position[0] -= self.cell_size
            self.current_mouse_image = self.mouse_images["left"]
        elif action == 3:  # Right
            self.agent_position[0] += self.cell_size
            self.current_mouse_image = self.mouse_images["right"]

        self.agent_position = np.clip(
            self.agent_position, 0, self.grid_size * self.cell_size - self.cell_size
        )

        # Calculate the relative position of the cheese
        cheese_relative_x = (
            self.reward_position[0] - self.agent_position[0]
        ) // self.cell_size
        cheese_relative_y = (
            self.reward_position[1] - self.agent_position[1]
        ) // self.cell_size

        # Check if the cheese is within the vision range
        if (
            abs(cheese_relative_x) <= self.vision_range
            and abs(cheese_relative_y) <= self.vision_range
        ):
            next_obs = (
                cheese_relative_x // env.cell_size,
                cheese_relative_y // env.cell_size,
            )

        else:
            next_obs = ("out_of_vision",)

        done = np.all(self.agent_position == self.reward_position)
        reward = 1 if done else -0.1

        return next_obs, reward, done, {}

    def reset(self):
        self.agent_position = (
            np.random.randint(0, self.grid_size * self.cell_size, 2)
            // self.cell_size
            * self.cell_size
        )
        self.reward_position = (
            np.random.randint(0, self.grid_size * self.cell_size, 2)
            // self.cell_size
            * self.cell_size
        )
        self.start_time = time.time()

        return np.concatenate((self.agent_position, self.reward_position))

    def render(self, episode, num_died, num_rewarded):
        self.screen.fill((255, 255, 255))

        # Draw the current mouse image instead of the blue rectangle
        self.screen.blit(self.current_mouse_image, self.agent_position)

        # Draw the mouse's sight
        sight_surface = pygame.Surface(
            (
                self.vision_range * 2 * self.cell_size,
                self.vision_range * 2 * self.cell_size,
            ),
            pygame.SRCALPHA,
        )
        sight_surface.fill((0, 0, 255, 64))
        self.screen.blit(
            sight_surface,
            (
                self.agent_position[0]
                - (self.vision_range * self.cell_size)
                + self.cell_size / 2,
                self.agent_position[1]
                - (self.vision_range * self.cell_size)
                + self.cell_size / 2,
            ),
        )

        # Draw the cheese
        self.screen.blit(self.cheese_image, self.reward_position)

        # Draw boundary box
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 900, 900), 2)

        # Display timer, episode number, and died and rewarded counters at the top of the screen
        elapsed_time = time.time() - self.start_time
        timer_text = font.render(f"Time: {elapsed_time:.1f}s", True, (0, 0, 0))
        self.screen.blit(timer_text, (10, 10))

        episode_text = font.render(f"Episode: {episode}", True, (0, 0, 0))
        self.screen.blit(episode_text, (200, 10))

        died_text = font.render(f"Died: {num_died}", True, (0, 0, 0))
        self.screen.blit(died_text, (400, 10))

        rewarded_text = font.render(f"Rewarded: {num_rewarded}", True, (0, 0, 0))
        self.screen.blit(rewarded_text, (550, 10))

        pygame.display.flip()

    def close(self):
        pygame.quit()


# Q-learning parameters
alpha = 0.2
gamma = 0.99
epsilon = 2.0
min_epsilon = 0.01
epsilon_decay = 0.995
num_episodes = 100

env = GridEnvironment()

# Initialize the Q-table
state_size = (env.grid_size, env.grid_size, env.grid_size, env.grid_size)
num_died = 0
num_rewarded = 0

# Load the Q-table
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
else:
    q_table = {
        (*coord, action): 0
        for coord in itertools.product(
            range(state_size[0]), range(state_size[1]), ["out_of_vision"]
        )
        for action in range(env.action_space.n)
    }


for episode in range(num_episodes):
    start_time = time.time()
    done = False
    obs = env.reset()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        env.render(episode, num_died, num_rewarded)

        if np.array_equal(obs, np.array(["out_of_vision"])):
            q_values = [0] * env.action_space.n
        else:
            q_values = [q_table.get((*obs, a), 0) for a in range(env.action_space.n)]

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values)

        next_obs, reward, done, _ = env.step(action)

        if not np.array_equal(next_obs, np.array(["out_of_vision"])):
            next_obs = tuple(next_obs[i] // env.cell_size for i in range(2))

        # Update Q-values
        if np.array_equal(next_obs, np.array(["out_of_vision"])):
            next_q_values = [0] * env.action_space.n
        else:
            next_q_values = [
                q_table.get((*next_obs, a), 0) for a in range(env.action_space.n)
            ]
        q_key = (*obs, action)
        q_table[q_key] = q_table.get(q_key, 0) + alpha * (
            reward + gamma * np.max(next_q_values) - q_table.get(q_key, 0)
        )

        obs = next_obs

        if time.time() - start_time > env.time_limit:  # Reset after 10 seconds
            num_died += 1
            break

        if done:  # The reward was found
            num_rewarded += 1

        # pygame.time.delay(100)

    # Decay epsilon for the next episode
    epsilon = max(min_epsilon, epsilon_decay * epsilon)

# Save the learning
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print(f"Episodes: {episode + 1} | Rewards: {num_rewarded}")

env.close()
