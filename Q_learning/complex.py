import boto3
import numpy as np
import time

# Initialize boto3 clients
lambda_client = boto3.client('lambda')
cloudwatch_client = boto3.client('cloudwatch')

# Define environment parameters
memory_sizes = [128, 256, 512, 1024, 2048]
timeouts = [1, 2, 3, 4, 5]
concurrency_limits = [1, 2, 5, 10]  # Example concurrency limits
actions = ['increase_memory', 'decrease_memory', 'increase_timeout', 'decrease_timeout', 'increase_concurrency', 'decrease_concurrency']
num_states = len(memory_sizes) * len(timeouts) * len(concurrency_limits)
num_actions = len(actions)


# Q-table dimensions
Q_shape = (num_memory_sizes * num_timeouts * num_concurrency_limits * num_image_sizes *
           num_invocation_frequencies * num_input_data_sizes * num_cpu_utilization_levels *
           num_io_wait_times * num_network_bandwidth_levels * num_error_rates, 6)

# Initialize Q-table with zeros
Q = np.zeros((num_states, num_actions))

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Function to map (memory, timeout, concurrency) to a state index
def get_state_index(memory, timeout, concurrency):
    return memory_sizes.index(memory) * len(timeouts) * len(concurrency_limits) + timeouts.index(timeout) * len(concurrency_limits) + concurrency_limits.index(concurrency)

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)  # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit

# Function to invoke the Lambda function with a given configuration
def invoke_lambda(memory, timeout, concurrency, function_name):
    lambda_client.update_function_configuration(
        FunctionName=function_name,
        MemorySize=memory,
        Timeout=timeout,
        ReservedConcurrentExecutions=concurrency
    )
    
    time.sleep(5)  # Wait for the configuration update
    
    invoke_response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse'
    )
    
    time.sleep(30)  # Simulate delay for CloudWatch metrics
    
    return invoke_response['LogResult']

# Function to get cost and runtime from CloudWatch metrics
def get_metrics(function_name, memory, timeout):
    start_time = int(time.time()) - 60
    end_time = int(time.time())
    
    duration_response = cloudwatch_client.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='Duration',
        Dimensions=[
            {'Name': 'FunctionName', 'Value': function_name},
            {'Name': 'Resource', 'Value': f'{function_name}:{timeout}'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=['Average']
    )
    
    duration = duration_response['Datapoints'][0]['Average'] if duration_response['Datapoints'] else 0
    
    cost_response = cloudwatch_client.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='EstimatedCost',
        Dimensions=[
            {'Name': 'FunctionName', 'Value': function_name}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=['Average']
    )
    
    cost = cost_response['Datapoints'][0]['Average'] if cost_response['Datapoints'] else 0
    
    return cost, duration

# Function to take an action and return the new state and reward
def take_action(memory, timeout, concurrency, action, function_name):
    if action == 0 and memory < max(memory_sizes):  # Increase memory
        memory = memory_sizes[memory_sizes.index(memory) + 1]
    elif action == 1 and memory > min(memory_sizes):  # Decrease memory
        memory = memory_sizes[memory_sizes.index(memory) - 1]
    elif action == 2 and timeout < max(timeouts):  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > min(timeouts):  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    elif action == 4 and concurrency < max(concurrency_limits):  # Increase concurrency
        concurrency = concurrency_limits[concurrency_limits.index(concurrency) + 1]
    elif action == 5 and concurrency > min(concurrency_limits):  # Decrease concurrency
        concurrency = concurrency_limits[concurrency_limits.index(concurrency) - 1]
    
    invoke_lambda(memory, timeout, concurrency, function_name)
    cost, runtime = get_metrics(function_name, memory, timeout)
    
    reward = - (cost + runtime)  # Negative reward to minimize cost and runtime
    
    new_state = get_state_index(memory, timeout, concurrency)
    
    return new_state, reward

# Training the Q-learning agent
num_episodes = 1000
lambda_functions = ['LambdaFunction1', 'LambdaFunction2', 'LambdaFunction3']  # Example Lambda functions

for episode in range(num_episodes):
    function_name = np.random.choice(lambda_functions)
    memory = np.random.choice(memory_sizes)
    timeout = np.random.choice(timeouts)
    concurrency = np.random.choice(concurrency_limits)
    state = get_state_index(memory, timeout, concurrency)
    
    for _ in range(100):  # Limit the number of steps per episode
        action = choose_action(state)
        new_state, reward = take_action(memory, timeout, concurrency, action, function_name)
        
        # Update Q-value
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        state = new_state

# Predicting optimal configuration for a new function
initial_memory = 128
initial_timeout = 1
initial_concurrency = 1
function_name = 'NewLambdaFunction'
state = get_state_index(initial_memory, initial_timeout, initial_concurrency)

for _ in range(100):  # Limit the number of steps
    action = choose_action(state)
    new_state, reward = take_action(initial_memory, initial_timeout, initial_concurrency, action, function_name)
    state = new_state

optimal_memory = memory_sizes[state // (len(timeouts) * len(concurrency_limits))]
optimal_timeout = timeouts[(state // len(concurrency_limits)) % len(timeouts)]
optimal_concurrency = concurrency_limits[state % len(concurrency_limits)]

print(f"Optimal configuration: Memory = {optimal_memory} MB, Timeout = {optimal_timeout} seconds, Concurrency = {optimal_concurrency}")
