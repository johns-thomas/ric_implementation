import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.fc(state)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.fc(state)

# Simulated environment
class ServerlessEnv:
    def __init__(self):
        # Example: 2-dimensional state space and 2-dimensional action space
        self.state_dim = 2
        self.action_dim = 2
    
    def step(self, action):
        # Simulate environment response to action
        # Here, we just use a random example
        next_state = np.random.rand(self.state_dim)
        cost = np.random.rand()  # Simulate cost
        runtime = np.random.rand()  # Simulate runtime
        reward = -(0.5 * cost + 0.5 * runtime)  # Example reward function
        return next_state, reward
    
    def reset(self):
        # Reset to initial configuration
        return np.random.rand(self.state_dim)

# Select action based on action probabilities
def select_action(action_probs):
    return np.random.choice(len(action_probs), p=action_probs.detach().numpy())

# Hyperparameters
state_dim = 2
action_dim = 2
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
num_episodes = 1000
max_timesteps = 100

# Initialize environment, actor, and critic
env = ServerlessEnv()
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = torch.FloatTensor(state)
    
    for t in range(max_timesteps):
        action_probs = actor(state)
        action = select_action(action_probs)
        next_state, reward = env.step(action)
        
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        
        # Calculate TD target and TD error
        td_target = reward + gamma * critic(next_state)
        td_error = td_target - critic(state)
        
        # Update Critic
        critic_loss = td_error.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update Actor
        actor_loss = -torch.log(action_probs[action]) * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        state = next_state
        
        if t == max_timesteps - 1:
            print(f"Episode {episode}, Timesteps {t}, Reward: {reward.item()}")
            break
