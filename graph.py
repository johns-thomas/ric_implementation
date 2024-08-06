import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('./formatted_dqn_results.json', 'r') as f:
    data = json.load(f)

# Extract data for analysis
episodes = [d['episodes'] for d in data]
execution_times = [np.mean(d['execution_times']) for d in data]
memory_usages = [np.mean(d['memory_usages']) for d in data]
costs = [np.mean(d['costs']) for d in data]
rewards = [np.mean(d['rewards']) for d in data]
memory_configs = [np.mean(d['memory_configurations']) for d in data]
timeout_configs = [np.mean(d['timeout_configurations']) for d in data]

# Plot Execution Time Trends
plt.figure(figsize=(8, 6))
plt.plot(episodes, execution_times, marker='o', linestyle='-')
plt.title('Average Execution Time per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Execution Time (s)')
plt.grid(True)
plt.show()

# Plot Memory Usage Trends
plt.figure(figsize=(12, 6))
plt.plot(episodes, memory_usages, marker='o', linestyle='-')
plt.title('Average Memory Usage per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Memory Usage (MB)')
plt.grid(True)
plt.show()

# Plot Cost Analysis
plt.figure(figsize=(12, 6))
plt.plot(episodes, costs, marker='o', linestyle='-')
plt.title('Average Cost per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Cost')
plt.grid(True)
plt.show()

# Plot Reward Patterns
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, marker='o', linestyle='-')
plt.title('Average Reward per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.grid(True)
plt.show()

# Plot Memory Configuration Impact
plt.figure(figsize=(12, 6))
plt.plot(episodes, memory_configs, marker='o', linestyle='-')
plt.title('Average Memory Configuration per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Memory Configuration (units)')
plt.grid(True)
plt.show()

# Plot Timeout Configuration Impact
plt.figure(figsize=(12, 6))
plt.plot(episodes, timeout_configs, marker='o', linestyle='-')
plt.title('Average Timeout Configuration per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Timeout Configuration (s)')
plt.grid(True)
plt.show()
