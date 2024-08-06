import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('./formatted_results.json', 'r') as f:
    data1 = json.load(f)
# Load the data
with open('./formatted_dqn_results.json', 'r') as f:
    data2 = json.load(f)
'''
# Extract data for analysis
episodes1 = [d['episodes'] for d in data1]
execution_times1 = [np.mean(d['execution_times']) for d in data1]
memory_usages1 = [np.mean(d['memory_usages']) for d in data1]
costs1 = [np.mean(d['costs']) for d in data1]
rewards1 = [np.mean(d['rewards']) for d in data1]
memory_configs1= [np.mean(d['memory_configurations']) for d in data1]
timeout_configs1 = [np.mean(d['timeout_configurations']) for d in data1]

# Extract data for analysis
episodes2 = [d['episodes'] for d in data2]
execution_times2 = [np.mean(d['execution_times']) for d in data2]
memory_usages2 = [np.mean(d['memory_usages']) for d in data2]
costs2 = [np.mean(d['costs']) for d in data2]
rewards2 = [np.mean(d['rewards']) for d in data2]
memory_configs2 = [np.mean(d['memory_configurations']) for d in data2]
timeout_configs2 = [np.mean(d['timeout_configurations']) for d in data2]
'''
episodes1 = [d['episodes'] for d in data1]
execution_times1 = [time for d in data1 for time in d['execution_times']]
memory_usages1 = [usage for d in data1 for usage in d['memory_usages']]

memory_configs1 = [config for d in data1 for config in d['memory_configurations']]
timeout_configs1 = [timeout for d in data1 for timeout in d['timeout_configurations']]
episodes2 = [d['episodes'] for d in data2]
execution_times2 = [time for d in data2 for time in d['execution_times']]
memory_usages2 = [usage for d in data2 for usage in d['memory_usages']]
costs2 = [cost for d in data2 for cost in d['costs']]
rewards2 = [reward for d in data2 for reward in d['rewards']]
memory_configs2 = [config for d in data2 for config in d['memory_configurations']]
timeout_configs2 = [timeout for d in data2 for timeout in d['timeout_configurations']]


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(memory_configs2, execution_times2, color='g', marker='o',label='DQN Agent')
plt.scatter(memory_configs1, execution_times1, color='b', marker='o',label='Q learning Agent')
plt.title('Execution Duration vs Memory Configuration- DQN Agent')
plt.xlabel('Memory Configuration (MB)')
plt.ylabel('Execution Duration (ms)')
plt.grid(True)
plt.legend()
plt.show()
'''
plt.figure(figsize=(10, 6))
plt.scatter(memory_configs1, execution_times1, color='g', marker='o')
plt.title('Execution Duration vs Memory Configuration- Q learning Agent')
plt.xlabel('Memory Configuration (MB)')
plt.ylabel('Execution Duration (ms)')
plt.grid(True)
plt.show()



# Plot Memory Usage Trends
plt.figure(figsize=(12, 6))
plt.plot(episodes1, memory_usages1, marker='o', linestyle='-',label='Max Memory used')
plt.plot(episodes1, memory_configs1, marker='o', linestyle='-',label='Memory Configured')
plt.title('Max Memory used vs Memory Configured- Q-learning Agent')
plt.xlabel('Episodes')
plt.ylabel('Memory(MB)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(episodes2, memory_usages2, marker='o', linestyle='-',label='Max Memory used')
plt.plot(episodes2, memory_configs2, marker='o', linestyle='-',label='Memory Configured')
plt.title('Max Memory used vs Memory Configured- DQN Agent')
plt.xlabel('Episodes')
plt.ylabel('Memory(MB)')
plt.grid(True)
plt.legend()
plt.show()

# Plot Cost Analysis
plt.figure(figsize=(12, 6))
plt.plot(episodes1, costs1, marker='o', linestyle='-',label='Q-learning Agent')
plt.plot(episodes2, costs2, marker='o', linestyle='-',label='DQN Agent')
plt.title('Average Cost per Episode')
plt.xlabel('Episodes')
plt.ylabel('Average Cost')
plt.grid(True)
plt.legend()
plt.show()
'''