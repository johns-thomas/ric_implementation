import numpy as np

# Define the environment parameters
memory_sizes = [128, 256, 512, 1024, 2048]  # Example memory sizes in MB
timeouts = [1, 2, 3, 4, 5]  # Example timeouts in seconds
actions = ['increase_memory', 'decrease_memory', 'increase_timeout', 'decrease_timeout']
num_states = len(memory_sizes) * len(timeouts)
num_actions = len(actions)

# Initialize Q-table with zeros
Q = np.zeros((num_states, num_actions))

# Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration factor

# Function to map (memory, timeout) to a state index
def get_state_index(memory, timeout):
    return memory_sizes.index(memory) * len(timeouts) + timeouts.index(timeout)

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)  # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit

# Function to take an action and return the new state and reward
def take_action(memory, timeout, action):
    if action == 0 and memory < max(memory_sizes):  # Increase memory
        memory = memory_sizes[memory_sizes.index(memory) + 1]
    elif action == 1 and memory > min(memory_sizes):  # Decrease memory
        memory = memory_sizes[memory_sizes.index(memory) - 1]
    elif action == 2 and timeout < max(timeouts):  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > min(timeouts):  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    
    # Simulate cost and run-time (replace with actual function call in real implementation)
    cost = 0.01 * memory  # Example cost function
    runtime = timeout - (0.001 * memory)  # Example runtime function
    
    reward = - (cost + runtime)  # Negative reward to minimize cost and runtime
    
    new_state = get_state_index(memory, timeout)
    
    return new_state, reward

# Training the Q-learning agent
num_episodes = 1000
for episode in range(num_episodes):
    memory = np.random.choice(memory_sizes)
    timeout = np.random.choice(timeouts)
    state = get_state_index(memory, timeout)
    
    for _ in range(100):  # Limit the number of steps per episode
        action = choose_action(state)
        new_state, reward = take_action(memory, timeout, action)
        
        # Update Q-value
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        state = new_state

# Predicting optimal configuration for a new function
initial_memory = 128  # Starting memory
initial_timeout = 1  # Starting timeout
state = get_state_index(initial_memory, initial_timeout)

for _ in range(100):  # Limit the number of steps
    action = choose_action(state)
    new_state, reward = take_action(initial_memory, initial_timeout, action)
    state = new_state

optimal_memory = memory_sizes[state // len(timeouts)]
optimal_timeout = timeouts[state % len(timeouts)]

print(f"Optimal configuration: Memory = {optimal_memory} MB, Timeout = {optimal_timeout} seconds")
