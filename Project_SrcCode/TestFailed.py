import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# GridWorld Environment (with moving obstacles)
# ---------------------------
class GridWorld:
    def __init__(self, grid_size=20, cell_size=30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_obstacles = int(grid_size * grid_size * 0.15)

        # fixed safe cells
        self.start_cell = [0, 0]
        self.goal_cell = [grid_size - 1, grid_size - 1]

        self._generate_world()
        self.reset()

    # -----------------------------------
    # Check if position is in protected zone
    # -----------------------------------
    def _is_protected(self, pos):
        x, y = pos
        # Top-left 4x4 zone (start area) - open on right side (x=3) and bottom side (y=3)
        if x < 4 and y < 4:
            if x == 3 or y == 3:  # right and bottom edges are open
                return False
            return True
        # Bottom-right 4x4 zone (goal area) - open on left side (x=grid_size-4) and top side (y=grid_size-4)
        if x >= self.grid_size - 4 and y >= self.grid_size - 4:
            if x == self.grid_size - 4 or y == self.grid_size - 4:  # left and top edges are open
                return False
            return True
        return False

    # -----------------------------------
    # Generate obstacles AWAY from start/goal and protected zones
    # -----------------------------------
    def _generate_world(self):
        self.obstacles = []

        while len(self.obstacles) < self.num_obstacles:
            pos = [random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1)]

            if pos not in self.obstacles and not self._is_protected(pos):
                self.obstacles.append(pos)

        # Give each obstacle a random movement direction
        self.obstacle_vel = []
        for _ in range(len(self.obstacles)):
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            self.obstacle_vel.append([dx, dy])

        self.goal_pos = self.goal_cell.copy()

    def reset(self):
        self.agent_pos = self.start_cell.copy()
        return tuple(self.agent_pos)

    # -----------------------------------
    # Obstacle movement (never enters protected zones)
    # -----------------------------------
    def update_obstacles(self):
        new_positions = []

        for i, (ox, oy) in enumerate(self.obstacles):
            dx, dy = self.obstacle_vel[i]
            nx, ny = ox + dx, oy + dy

            # Bounce off walls
            if nx < 0 or nx >= self.grid_size:
                dx = -dx
                nx = ox + dx
            if ny < 0 or ny >= self.grid_size:
                dy = -dy
                ny = oy + dy

            # Save updated velocity
            self.obstacle_vel[i] = [dx, dy]

            # Prevent obstacle from entering protected zones
            if self._is_protected([nx, ny]):
                nx, ny = ox, oy  # stay put

            new_positions.append([nx, ny])

        self.obstacles = new_positions

    # -----------------------------------
    # Agent step + obstacle movement
    # -----------------------------------
    def step(self, action):
        x, y = self.agent_pos
        next_x, next_y = x, y

        # Agent movement
        if action == 'up': next_y = max(y - 1, 0)
        elif action == 'down': next_y = min(y + 1, self.grid_size - 1)
        elif action == 'left': next_x = max(x - 1, 0)
        elif action == 'right': next_x = min(x + 1, self.grid_size - 1)

        done = False

        # If agent walks INTO a static obstacle
        if [next_x, next_y] in self.obstacles:
            reward = -10
        else:
            self.agent_pos = [next_x, next_y]

            # Check goal
            if self.agent_pos == self.goal_cell:
                reward = 100
                done = True
            else:
                reward = -1

        # Now move obstacles
        self.update_obstacles()

        # If an obstacle MOVED INTO the agent
        if self.agent_pos in self.obstacles:
            reward = -20
            done = True

        return tuple(self.agent_pos), reward, done

    # -----------------------------------
    # Rendering
    # -----------------------------------
    def render(self, screen, path=None, agent_pos=None):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)

                # grid
                pygame.draw.rect(screen, (30, 30, 30), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)

                # highlight protected zones with subtle green tint
                if self._is_protected([x, y]):
                    pygame.draw.rect(screen, (10, 40, 10), rect)
                    pygame.draw.rect(screen, (50, 50, 50), rect, 1)

                # obstacles
                if [x, y] in self.obstacles:
                    pygame.draw.rect(screen, (0, 0, 255), rect)

                # goal
                if [x, y] == self.goal_cell:
                    pygame.draw.rect(screen, (0, 255, 0), rect)

                # path render
                if path and (x, y) in path:
                    pygame.draw.rect(screen, (255, 255, 0), rect)

                # agent
                if agent_pos and list(agent_pos) == [x, y]:
                    pygame.draw.rect(screen, (255, 0, 0), rect)


# ---------------------------
# Q-Learning Agent
# ---------------------------
class QLearningAgent:
    def __init__(self, grid_size=20, actions=['up','down','left','right'],
                 alpha=0.1, gamma=0.95, epsilon=1.0):
        self.grid_size = grid_size
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.Q = np.zeros((grid_size, grid_size, len(actions)))

    def choose_action(self, state):
        x, y = state
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            qvals = self.Q[x, y]
            best = np.argwhere(qvals == np.max(qvals)).flatten()
            return self.actions[random.choice(best)]

    def learn(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        ai = self.actions.index(action)

        target = reward + self.gamma * np.max(self.Q[nx, ny])
        self.Q[x, y, ai] += self.alpha * (target - self.Q[x, y, ai])


# ---------------------------
# Plot learning curve
# ---------------------------
def plot_learning_curve(rewards, lengths, success):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0][0].plot(rewards)
    axes[0][0].set_title("Rewards")

    axes[0][1].plot(lengths)
    axes[0][1].set_title("Episode Lengths")

    axes[1][0].plot(success)
    axes[1][0].set_title("Success Rate (%)")

    axes[1][1].axis('off')

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()


# ---------------------------
# Generate greedy path
# ---------------------------
def generate_learned_path(env, agent):
    state = env.reset()
    path = [state]
    for _ in range(500):
        x, y = state
        action = agent.actions[np.argmax(agent.Q[x, y])]

        nx, ny = x, y
        if action == 'up': ny = max(y - 1, 0)
        elif action == 'down': ny = min(y + 1, env.grid_size - 1)
        elif action == 'left': nx = max(x - 1, 0)
        elif action == 'right': nx = min(x + 1, env.grid_size - 1)

        if [nx, ny] in env.obstacles:
            break

        state = (nx, ny)
        path.append(state)

        if state == tuple(env.goal_cell):
            break
    return path


# ---------------------------
# Main loop (frame-by-frame training)
# ---------------------------
def main():
    GRID = 20
    SIZE = 30
    W = GRID * SIZE

    pygame.init()
    screen = pygame.display.set_mode((W, W))
    pygame.display.set_caption("Q-Learning with Protected 4x4 Zones")
    clock = pygame.time.Clock()

    env = GridWorld(GRID, SIZE)
    agent = QLearningAgent(GRID)

    episodes = 500
    max_steps = 300

    rewards = []
    lengths = []
    success = []
    sc = 0

    for ep in range(episodes):
        state = env.reset()
        total = 0
        done = False
        step = 0

        while not done and step < max_steps:
            step += 1
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            total += reward
            state = next_state

            # ---- FRAME-BY-FRAME RENDER ----
            screen.fill((0,0,0))
            env.render(screen, agent_pos=state)
            f = pygame.font.Font(None, 28)
            text = f.render(f"EP {ep+1} | STEP {step} | R={reward}", True, (255,255,255))
            screen.blit(text, (10,10))
            pygame.display.flip()
            clock.tick(10)

        # metrics
        rewards.append(total)
        lengths.append(step)
        if done and total > 0:
            sc += 1
        success.append(sc / (ep + 1) * 100)

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    pygame.quit()

    plot_learning_curve(rewards, lengths, success)


if __name__ == "__main__":
    main()