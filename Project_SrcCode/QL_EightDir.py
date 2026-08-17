import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# GridWorld Environment
# ---------------------------
class GridWorld:
    def __init__(self, grid_size=20, cell_size=30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_obstacles = int(grid_size * grid_size * 0.15)
        self._generate_world()
        self.reset()

    def _generate_world(self):
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            pos = [random.randint(0, self.grid_size-1),
                   random.randint(0, self.grid_size-1)]
            if pos not in self.obstacles and pos != [0,0] and pos != [self.grid_size-1,self.grid_size-1]:
                self.obstacles.append(pos)
        self.goal_pos = [self.grid_size-1, self.grid_size-1]

    def reset(self):
        self.agent_pos = [0,0]
        return tuple(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        next_x, next_y = x, y
        
        # 8-directional movement
        if action == 'up': 
            next_y = max(y-1, 0)
        elif action == 'down': 
            next_y = min(y+1, self.grid_size-1)
        elif action == 'left': 
            next_x = max(x-1, 0)
        elif action == 'right': 
            next_x = min(x+1, self.grid_size-1)
        elif action == 'up-left':
            next_x = max(x-1, 0)
            next_y = max(y-1, 0)
        elif action == 'up-right':
            next_x = min(x+1, self.grid_size-1)
            next_y = max(y-1, 0)
        elif action == 'down-left':
            next_x = max(x-1, 0)
            next_y = min(y+1, self.grid_size-1)
        elif action == 'down-right':
            next_x = min(x+1, self.grid_size-1)
            next_y = min(y+1, self.grid_size-1)
        
        done = False
        
        # Check if hit obstacle
        if [next_x, next_y] in self.obstacles:
            reward = -10
            # Stay in place when hitting obstacle
        else:
            # Valid move
            self.agent_pos = [next_x, next_y]
            # Check if reached goal
            if self.agent_pos == self.goal_pos:
                reward = 100  # Goal reward
                done = True
            else:
                # Diagonal moves cost slightly more (realistic distance)
                if action in ['up-left', 'up-right', 'down-left', 'down-right']:
                    reward = -1.4  # sqrt(2) ≈ 1.414
                else:
                    reward = -1
        
        return tuple(self.agent_pos), reward, done

    def render(self, screen, path=None, agent_pos=None):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                                   self.cell_size, self.cell_size)
                # Draw grid background
                pygame.draw.rect(screen, (30,30,30), rect)
                pygame.draw.rect(screen, (50,50,50), rect, 1)
                
                # Draw obstacles (blue)
                if [x,y] in self.obstacles: 
                    pygame.draw.rect(screen, (0,0,255), rect)
                
                # Draw goal (green)
                if [x,y] == self.goal_pos: 
                    pygame.draw.rect(screen, (0,255,0), rect)
                
                # Draw path (yellow)
                if path and (x,y) in path: 
                    pygame.draw.rect(screen, (255,255,0), rect)
                
                # Draw agent (red)
                if agent_pos and list(agent_pos) == [x,y]: 
                    pygame.draw.rect(screen, (255,0,0), rect)

# ---------------------------
# Q-Learning Agent
# ---------------------------
class QLearningAgent:
    def __init__(self, grid_size=20, 
                 actions=['up', 'down', 'left', 'right', 
                         'up-left', 'up-right', 'down-left', 'down-right'],
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
            max_q = np.max(self.Q[x,y])
            max_actions = [i for i, q in enumerate(self.Q[x,y]) if q == max_q]
            return self.actions[random.choice(max_actions)]

    def learn(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        idx = self.actions.index(action)
        target = reward + self.gamma * np.max(self.Q[nx, ny])
        self.Q[x, y, idx] += self.alpha * (target - self.Q[x, y, idx])

# ---------------------------
# Generate path from Q-table
# ---------------------------
def generate_learned_path(env, agent):
    state = env.reset()
    path = [state]
    visited = set()
    visited.add(state)
    max_path_length = 1000
    
    while state != tuple(env.goal_pos) and len(path) < max_path_length:
        x, y = state
        
        # Get all Q-values for current state
        q_values = agent.Q[x, y].copy()
        
        # Try actions in order of Q-value (best first)
        action_indices = np.argsort(q_values)[::-1]
        moved = False
        
        for idx in action_indices:
            action = agent.actions[idx]
            nx, ny = x, y
            
            # Calculate next position based on action
            if action == 'up': 
                ny = max(y-1, 0)
            elif action == 'down': 
                ny = min(y+1, env.grid_size-1)
            elif action == 'left': 
                nx = max(x-1, 0)
            elif action == 'right': 
                nx = min(x+1, env.grid_size-1)
            elif action == 'up-left':
                nx = max(x-1, 0)
                ny = max(y-1, 0)
            elif action == 'up-right':
                nx = min(x+1, env.grid_size-1)
                ny = max(y-1, 0)
            elif action == 'down-left':
                nx = max(x-1, 0)
                ny = min(y+1, env.grid_size-1)
            elif action == 'down-right':
                nx = min(x+1, env.grid_size-1)
                ny = min(y+1, env.grid_size-1)
            
            # Skip if obstacle
            if [nx, ny] in env.obstacles: 
                continue
            
            # Skip if no actual movement
            if (nx, ny) == state:
                continue
                
            next_state = (nx, ny)
            
            # Prefer unvisited states, but allow revisiting if necessary
            if next_state not in visited or not moved:
                path.append(next_state)
                if next_state not in visited:
                    visited.add(next_state)
                state = next_state
                moved = True
                break
        
        if not moved:
            print("Warning: Agent got stuck before reaching goal")
            break
    
    if state == tuple(env.goal_pos):
        print(f"Success! Path reached goal in {len(path)} steps")
    else:
        print(f"Warning: Path did not reach goal. Ended at {state}, goal is {tuple(env.goal_pos)}")
    
    return path

# ---------------------------
# Plot learning curve
# ---------------------------
def plot_learning_curve(episode_rewards, episode_lengths, success_history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue')
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       color='red', linewidth=2, label=f'{window}-episode MA')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6, color='green')
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg, 
                       color='red', linewidth=2, label=f'{window}-episode MA')
        axes[0, 1].legend()
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    axes[1, 0].plot(success_history, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title(f'Success Rate (Window: {window} episodes)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    # Plot 4: Summary Statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Episodes: {len(episode_rewards)}
    Movement: 8-Directional
    
    Final Success Rate: {success_history[-1]:.1f}%
    
    Average Reward (last 100): 
    {np.mean(episode_rewards[-100:]):.2f}
    
    Average Length (last 100): 
    {np.mean(episode_lengths[-100:]):.2f}
    
    Best Episode Reward: {max(episode_rewards):.2f}
    Worst Episode Reward: {min(episode_rewards):.2f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('learning_curve_8dir.png', dpi=150, bbox_inches='tight')
    print("\nLearning curve saved as 'learning_curve_8dir.png'")
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    GRID_SIZE = 20
    CELL_SIZE = 30
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("GridWorld Q-Learning - 8-Directional Movement")
    clock = pygame.time.Clock()

    env = GridWorld(GRID_SIZE, CELL_SIZE)
    agent = QLearningAgent(GRID_SIZE)

    print("Training agent with 8-directional movement...")
    num_episodes = 2000
    max_steps = 500
    success_count = 0
    
    episode_rewards = []
    episode_lengths = []
    success_history = []

    #  Track visits for heatmap
    visit_map = np.zeros((GRID_SIZE, GRID_SIZE))

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < max_steps:
            step += 1

            visit_map[state[0], state[1]] += 1     # 🔴 Count visits per state

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        if done: success_count += 1
        
        # Success rate calculation
        window = 50
        if episode >= window:
            recent_success = sum(1 for i in range(episode - window + 1, episode + 1)
                                 if episode_lengths[i] < max_steps and episode_rewards[i] > 0)
            success_rate = (recent_success / window) * 100
        else:
            success_rate = (success_count / (episode + 1)) * 100
        success_history.append(success_rate)
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        if (episode + 1) % 200 == 0:
            print(f"Episode {episode+1}/{num_episodes}  |  Success: {success_rate:.1f}%  |  ε={agent.epsilon:.3f}")

    print(f"\nTraining complete! {success_count}/{num_episodes} successful episodes")
    print(f"Final epsilon: {agent.epsilon:.3f}")

    plot_learning_curve(episode_rewards, episode_lengths, success_history)

    # ================= SAVE HEATMAP =================
    plt.imshow(visit_map, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Visit Frequency")
    plt.title("State Visit Heatmap - 8 Direction Q-Learning")
    plt.savefig("visit_heatmap_8dir.png", dpi=200)
    plt.close()
    print("\nHeatmap saved as: visit_heatmap_8dir.png")

    print("\nGenerating final learned path...")
    path = generate_learned_path(env, agent)
    print("Animating path step-by-step...")

    # ================== ANIMATE PATH ================== #
    for i, pos in enumerate(path):
        screen.fill((0,0,0))
        env.render(screen, path=path[:i+1], agent_pos=pos)
        pygame.display.flip()
        pygame.time.delay(120)        # ← Change speed here

    # Window stays open until exit
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()