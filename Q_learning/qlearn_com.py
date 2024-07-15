import numpy as np
import boto3
from datetime import datetime, timedelta

# Define the ranges and categories
memory_sizes = [128, 256, 512, 1024, 2048, 3008]
timeouts = [1, 2, 3, 5, 10, 15]
cpu_utilization_levels = [0, 50, 80, 100]  # Low: 0-50%, Medium: 51-80%, High: 81-100%
image_size_categories = [0, 500000, 1000000, 5000000]  # Small: <500KB, Medium: 500KB-1MB, Large: 1MB-5MB, Extra-large: >5MB

# Q-table dimensions
Q_shape = (len(memory_sizes) * len(timeouts) * len(cpu_utilization_levels) * len(image_size_categories), 6)
Q = np.zeros(Q_shape)

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# Function to get the category for image size
def get_image_size_category(image_size):
    for i, threshold in enumerate(image_size_categories):
        if image_size <= threshold:
            return i
    return len(image_size_categories) - 1

# Function to get the category for CPU utilization
def get_cpu_utilization_category(cpu_utilization):
    for i, threshold in enumerate(cpu_utilization_levels):
        if cpu_utilization <= threshold:
            return i
    return len(cpu_utilization_levels) - 1

# Function to get state index
def get_state_index(memory, timeout, cpu_utilization, image_size_category):
    memory_index = memory_sizes.index(memory)
    timeout_index = timeouts.index(timeout)
    cpu_utilization_index = get_cpu_utilization_category(cpu_utilization)
    image_size_index = image_size_category

    return (memory_index * len(timeouts) * len(cpu_utilization_levels) * len(image_size_categories) +
            timeout_index * len(cpu_utilization_levels) * len(image_size_categories) +
            cpu_utilization_index * len(image_size_categories) +
            image_size_index)

# AWS Lambda and CloudWatch clients
lambda_client = boto3.client('lambda')
cloudwatch_client = boto3.client('cloudwatch')

# Function to get current CPU utilization from CloudWatch
def get_current_cpu_utilization(cloudwatch_client, function_name, period=60):
    namespace = 'AWS/Lambda'
    metric_name = 'Duration'  # This is used as a proxy for CPU utilization
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)

    response = cloudwatch_client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[
            {
                'Name': 'FunctionName',
                'Value': function_name
            }
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        Statistics=['Average']
    )
    
    data_points = response.get('Datapoints', [])
    if not data_points:
        return 0

    latest_data_point = max(data_points, key=lambda x: x['Timestamp'])
    cpu_utilization = latest_data_point['Average']

    return cpu_utilization

# Function to execute an action and get the new configuration
def execute_action(memory, timeout, action):
    if action == 0 and memory < memory_sizes[-1]:  # Increase memory
        memory = memory_sizes[memory_sizes.index(memory) + 1]
    elif action == 1 and memory > memory_sizes[0]:  # Decrease memory
        memory = memory_sizes[memory_sizes.index(memory) - 1]
    elif action == 2 and timeout < timeouts[-1]:  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > timeouts[0]:  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    elif action == 4:  # Reserved for future actions (e.g., concurrency adjustments)
        pass
    elif action == 5:  # Reserved for future actions (e.g., concurrency adjustments)
        pass

    return memory, timeout

# Training the RL agent
for episode in range(num_episodes):
    # Initialize state with random values
    memory = np.random.choice(memory_sizes)
    timeout = np.random.choice(timeouts)
    cpu_utilization = np.random.choice(cpu_utilization_levels)
    image_size = np.random.choice(range(len(image_size_categories)))
    
    state = get_state_index(memory, timeout, cpu_utilization, image_size)
    done = False
    
    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.choice(6)  # Exploration
        else:
            action = np.argmax(Q[state])  # Exploitation
        
        # Execute action and observe the result
        new_memory, new_timeout = execute_action(memory, timeout, action)
        image_size = get_current_image_size(lambda_client)  # Get current image size at runtime
        new_image_size_category = get_image_size_category(image_size)
        
        # Collect CPU utilization
        cpu_utilization = get_current_cpu_utilization(cloudwatch_client, 'your_lambda_function_name')
        new_cpu_utilization = get_cpu_utilization_category(cpu_utilization)
        
        new_state = get_state_index(new_memory, new_timeout, new_cpu_utilization, new_image_size_category)
        
        # Get cost and runtime from CloudWatch (dummy values used for illustration)
        cost = np.random.rand()  # Placeholder for actual cost retrieval
        runtime = np.random.rand()  # Placeholder for actual runtime retrieval
        
        # Calculate reward
        reward = - (cost + runtime)
        
        # Q-value update
        best_next_action = np.argmax(Q[new_state])
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[new_state, best_next_action] - Q[state, action])
        
        # Transition to new state
        memory, timeout = new_memory, new_timeout
        state = new_state
        
        # Define the terminal condition (e.g., fixed number of steps or satisfactory reward)
        done = np.random.rand() < 0.1  # Example condition, replace with your logic

# Save trained Q-table
np.save('q_table.npy', Q)

# Predicting the optimal configuration for a new image processing task
def predict_optimal_configuration(image_size, cpu_utilization):
    image_size_category = get_image_size_category(image_size)
    cpu_utilization_category = get_cpu_utilization_category(cpu_utilization)
    
    # Initialize state with default values
    memory = 512  # Example default
    timeout = 5
    
    state = get_state_index(memory, timeout, cpu_utilization_category, image_size_category)
    
    max_steps = 10  # Maximum steps to predict the configuration
    for step in range(max_steps):
        action = np.argmax(Q[state])  # Select the action with the highest Q-value
        
        # Execute action and transition to the new state
        new_memory, new_timeout = execute_action(memory, timeout, action)
        state = get_state_index(new_memory, new_timeout, cpu_utilization_category, image_size_category)
    
    return memory, timeout

# Example usage for a new task
new_image_size = 750000  # Example image size in bytes
new_cpu_utilization = 60  # Example CPU utilization

predicted_memory, predicted_timeout = predict_optimal_configuration(new_image_size, new_cpu_utilization)
print(f'Predicted optimal configuration: Memory={predicted_memory} MB, Timeout={predicted_timeout} s')
