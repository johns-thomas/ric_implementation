import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the data from the JSON file
with open('./formatted_results.json', 'r') as f:
    data = json.load(f)

# Extract rewards data from multiple episodes
rewards = [d['rewards'] for d in data]

# Calculate number of episodes
episodes = list(range(len(rewards)))


# Calculate statistical metrics for each episode
reward_iqr = [np.percentile(r, 75) - np.percentile(r, 25) for r in rewards]
reward_25th_percentile = [np.percentile(r, 25) for r in rewards]
reward_75th_percentile = [np.percentile(r, 75) for r in rewards]
reward_skewness = [stats.skew(r) for r in rewards]
reward_kurtosis = [stats.kurtosis(r) for r in rewards]

# Plot Interquartile Range (IQR)
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_iqr, marker='o', linestyle='-', label='IQR')
plt.title('Interquartile Range of Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('IQR')
plt.grid(True)
plt.legend()
plt.show()

# Plot 25th and 75th Percentiles
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_25th_percentile, marker='o', linestyle='-', label='25th Percentile')
plt.plot(episodes, reward_75th_percentile, marker='o', linestyle='-', label='75th Percentile')
plt.title('25th and 75th Percentiles of Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Percentile')
plt.grid(True)
plt.legend()
plt.show()

# Plot Skewness
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_skewness, marker='o', linestyle='-', label='Skewness')
plt.title('Skewness of Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Skewness')
plt.grid(True)
plt.legend()
plt.show()

# Plot Kurtosis
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_kurtosis, marker='o', linestyle='-', label='Kurtosis')
plt.title('Kurtosis of Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Kurtosis')
plt.grid(True)
plt.legend()
plt.show()
