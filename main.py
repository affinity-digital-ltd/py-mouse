import gymnasium as gym
import pygame
import numpy as np
import time
from gymnasium import spaces
from dqn_agent import DQNAgent
import random

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
            out_of_vision = 0
            next_obs = (
                cheese_relative_x // self.cell_size,
                cheese_relative_y // self.cell_size,
                out_of_vision,
            )

        else:
            out_of_vision = 1
            next_obs = (0, 0, out_of_vision)

        done = np.all(self.agent_position == self.reward_position)
        reward = 1 if done else -0.1

        return next_obs, reward, done, {}

    def reset(self):
        self.start_time = time.time()

        # Randomize initial positions
        self.agent_position = np.array(
            [random.randrange(0, self.grid_size, self.cell_size) for _ in range(2)]
        )
        self.reward_position = np.array(
            [random.randrange(0, self.grid_size, self.cell_size) for _ in range(2)]
        )

        # Initialize the mouse image based on the initial position
        self.current_mouse_image = self.mouse_images["up"]

        # Check if the agent is out of vision range
        distance = np.linalg.norm(self.agent_position - self.reward_position)
        out_of_vision = distance > self.vision_range
        return self.agent_position, self.reward_position, out_of_vision

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


env = GridEnvironment()

# DQN parameters
gamma = 0.95
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
num_episodes = 100
batch_size = 64
update_frequency = 5
num_died = 0
num_rewarded = 0

# Initialize the DQN agent
input_state_size = 5  # As you have 5 states including "out_of_vision"
action_size = env.action_space.n
agent = DQNAgent(input_state_size, action_size)

# Training loop
for episode in range(num_episodes):
    start_time = time.time()
    done = False
    agent_position, cheese_position, out_of_vision = env.reset()
    obs = np.array((*agent_position, *cheese_position, out_of_vision))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        env.render(episode, num_died, num_rewarded)

        action = agent.choose_action(obs)
        next_obs, reward, done, _ = env.step(action)
        next_obs = np.reshape(
            next_obs, [1, input_state_size]
        )  # Use input_state_size here

        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if time.time() - start_time > env.time_limit:  # Reset after 10 seconds
            num_died += 1
            break

        if done:  # The reward was found
            num_rewarded += 1

    # Decay epsilon for the next episode
    agent.epsilon = max(min_epsilon, epsilon_decay * agent.epsilon)

# Save the final model
agent.save("dqn_weights_final.h5")

print("Training complete.")

# Test the trained agent
agent.load("dqn_weights_final.h5")
agent.epsilon = 0  # Disable exploration

for episode in range(10):
    state = env.reset()
    state = np.reshape(state, [1, input_state_size])
    done = False
    step = 0

    while not done:
        env.render(
            episode, num_died, num_rewarded
        )  # Update render method with appropriate arguments
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, input_state_size])
        state = next_state
        step += 1

        if done:
            print(f"Test Episode: {episode + 1}, Score: {step}")
            break

print(f"Episodes: {episode + 1} | Rewards: {num_rewarded}")

env.close()
