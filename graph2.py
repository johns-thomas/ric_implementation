import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('./formatted_results.json', 'r') as f:
    data = json.load(f)
# Sample rewards data from multiple episodes
# Extract data for analysis
episodes = [d['episodes'] for d in data]
execution_times = [np.mean(d['execution_times']) for d in data]
memory_usages = [np.mean(d['memory_usages']) for d in data]
costs = [np.mean(d['costs']) for d in data]
rewards = [np.mean(d['rewards']) for d in data]
memory_configs = [np.mean(d['memory_configurations']) for d in data]
timeout_configs = [np.mean(d['timeout_configurations']) for d in data]

# Calculate metrics
reward_median = [np.median(r) for r in rewards]
reward_std_dev = [np.std(r) for r in rewards]
reward_25th_percentile = [np.percentile(r, 25) for r in rewards]
reward_75th_percentile = [np.percentile(r, 75) for r in rewards]
reward_min = [np.min(r) for r in rewards]
reward_max = [np.max(r) for r in rewards]
reward_iqr = [np.percentile(r, 75) - np.percentile(r, 25) for r in rewards]
reward_variance = [np.var(r) for r in rewards]
reward_skewness = [np.mean((r - np.mean(r))**3) / np.std(r)**3 for r in rewards]
reward_kurtosis = [np.mean((r - np.mean(r))**4) / np.std(r)**4 for r in rewards]
reward_cv = [np.std(r) / np.mean(r) for r in rewards if np.mean(r) != 0]

# Example plot of median rewards
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_median, marker='o', linestyle='-')
plt.title('Median Reward per Episode')
plt.xlabel('Episodes')
plt.ylabel('Median Reward')
plt.grid(True)
plt.show()

# Example plot of standard deviation of rewards
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_std_dev, marker='o', linestyle='-')
plt.title('Reward Standard Deviation per Episode')
plt.xlabel('Episodes')
plt.ylabel('Standard Deviation of Rewards')
plt.grid(True)
plt.show()
