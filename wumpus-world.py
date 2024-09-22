import pygame
import random
import numpy as np
import time
import sys

# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 4
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GOLD_COLOR = (255, 215, 0)
GREEN = (0, 255, 0)
BORDER_COLOR = (0, 0, 255)

# Initialize the screen
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Wumpus World")

# Load images
agent_img = pygame.image.load('agent.png')
wumpus_img = pygame.image.load('wumpus.png')
pit_img = pygame.image.load('pit.png')
gold_img = pygame.image.load('gold.png')

# Scale images to fit the grid cells
agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
wumpus_img = pygame.transform.scale(wumpus_img, (CELL_SIZE, CELL_SIZE))
pit_img = pygame.transform.scale(pit_img, (CELL_SIZE, CELL_SIZE))
gold_img = pygame.transform.scale(gold_img, (CELL_SIZE, CELL_SIZE))

# Define positions
agent_start_pos = [0, 0]
breez_pos = [(0, 1), (1, 0), (2, 1), (1, 2), (1, 3)]
glitter_pos = [(2, 3), (2, 2)]
stinch_pos = [(2, 1), (3, 2), (2, 3), (1, 2)]
gold_pos = (1, 3)
wumpus_pos = (2, 2)
pit_positions = [(1, 1), (2, 3)]

# Define actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
action_map = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# Q-learning parameters
epsilon = 0.1  # Exploration rate
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))  # Q-table


def draw_grid():
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BORDER_COLOR, rect, 1)  # Draw the grid border with the border color

def draw_objects(agent_pos):
    # Draw pits
    for pit in pit_positions:
        screen.blit(pit_img, (pit[0] * CELL_SIZE, pit[1] * CELL_SIZE))

    # Draw Wumpus
    screen.blit(wumpus_img, (wumpus_pos[0] * CELL_SIZE, wumpus_pos[1] * CELL_SIZE))

    # Draw gold
    screen.blit(gold_img, (gold_pos[0] * CELL_SIZE, gold_pos[1] * CELL_SIZE))

    # Draw agent
    screen.blit(agent_img, (agent_pos[0] * CELL_SIZE, agent_pos[1] * CELL_SIZE))

def get_state(agent_pos):
    return agent_pos[0], agent_pos[1]

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(q_table[state[0], state[1]])]

def move_agent(agent_pos, action):
    move = action_map[action]
    new_pos = [agent_pos[0] + move[0], agent_pos[1] + move[1]]
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
        return new_pos
    return agent_pos

def get_reward(agent_pos):
    if tuple(agent_pos) in pit_positions:
        return -10
    elif tuple(agent_pos) == wumpus_pos:
        return -10
    elif tuple(agent_pos) == gold_pos:
        return 10
    else:
        return -1

def check_game_over(agent_pos):
    return tuple(agent_pos) in pit_positions or tuple(agent_pos) == wumpus_pos or tuple(agent_pos) == gold_pos

def visualize_training(agent_pos, episode, step):
    screen.fill(WHITE)
    draw_grid()
    draw_objects(agent_pos)
    pygame.display.flip()
    # time.sleep(0.7)  # Adding a delay to visualize the training process
    print(f"Episode {episode + 1}, Step {step + 1}")

# Main loop
for episode in range(100):  # Training for 15 episodes
    agent_pos = agent_start_pos.copy()
    state = get_state(agent_pos)
    total_reward = 0

    for step in range(100):  # Limiting the number of steps per episode
        action = choose_action(state)
        new_pos = move_agent(agent_pos, action)
        reward = get_reward(new_pos)
        total_reward += reward
        new_state = get_state(new_pos)
        
        # Update Q-value
        best_next_action = np.argmax(q_table[new_state[0], new_state[1]])
        td_target = reward + gamma * q_table[new_state[0], new_state[1], best_next_action]
        td_error = td_target - q_table[state[0], state[1], ACTIONS.index(action)]
        q_table[state[0], state[1], ACTIONS.index(action)] += alpha * td_error
        # print(q_table)
        agent_pos = new_pos
        state = new_state

        visualize_training(agent_pos, episode, step)
        
        if check_game_over(agent_pos):
            break

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Testing the agent
running = True
agent_pos = agent_start_pos.copy()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state(agent_pos)
    action = ACTIONS[np.argmax(q_table[state[0], state[1]])]
    agent_pos = move_agent(agent_pos, action)
    
    screen.fill(WHITE)
    draw_grid()
    draw_objects(agent_pos)
    pygame.display.flip()
    
    if check_game_over(agent_pos):
        print("Game Over!")
        print(q_table)
        running = False
        time.sleep(2)

    pygame.event.pump()  # Handle pygame events

pygame.quit()
