import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
from collections import deque
import boto3
import json
import time
from datetime import datetime, timedelta
import helper 
# Define the environment and DQN parameters
memory_sizes = np.arange(128, 3009)  # Continuous memory sizes from 128MB to 3008MB
timeouts = [1, 2, 3, 5, 10, 15]      # Discrete timeout values
duration_categories = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]  # Updated duration categories

state_size = 3  # memory, timeout, duration_category
action_size = 4  # increase memory, decrease memory, increase timeout, decrease timeout

# DQN parameters
gamma = 0.9
epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_buffer = deque(maxlen=2000)
num_episodes = 5
max_steps_per_episode = 10

# Define the DQN model
def build_model(state_size, action_size):
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# Initialize the model
model = build_model(state_size, action_size)
target_model = build_model(state_size, action_size)
target_model.set_weights(model.get_weights())

# Function to get the duration category
def get_duration_category(duration):
    for i, threshold in enumerate(duration_categories):
        if duration <= threshold:
            return i
    return len(duration_categories) - 1

# Function to get the CloudWatch logs
def get_cloudwatch_logs(log_group_name, start_time, end_time):
    logs_client = boto3.client('logs')
    response = logs_client.filter_log_events(
        logGroupName=log_group_name,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000),
        limit=1,
        interleaved=True
    )
    if 'events' in response and response['events']:
        log_event = response['events'][0]['message']
        return log_event
    return None

# Function to parse CloudWatch log events
def parse_log_event(log_event):
    parts = log_event.split('\t')
    metrics = {'out_of_memory': False, 'timed_out': False}
    for part in parts:
        if 'Duration' in part:
            metrics['Duration'] = float(part.split(': ')[1].replace(' ms', ''))
        elif 'Billed Duration' in part:
            metrics['Billed Duration'] = float(part.split(': ')[1].replace(' ms', ''))
        elif 'Memory Size' in part:
            metrics['Memory Size'] = int(part.split(': ')[1].replace(' MB', ''))
        elif 'Max Memory Used' in part:
            metrics['Max Memory Used'] = int(part.split(': ')[1].replace(' MB', ''))
        if 'Task timed out' in part:
            metrics['timed_out'] = True
        if 'fatal error' in part and 'Cannot allocate memory' in part:
            metrics['out_of_memory'] = True
    return metrics

def calculate_cost(memory, duration):
    memory = int(memory) 
    duration = float(duration) 
    return (memory / 1024) * duration * 0.00001667
# Function to calculate the reward
def calculate_reward(metrics, new_memory, new_timeout):
    if metrics['out_of_memory']:
        return -100  # Large penalty for out-of-memory errors
    if metrics['timed_out']:
        return -50  # Penalty for timeouts
    # Reward based on performance and resource usage
    cost = calculate_cost(new_memory, metrics['Duration'])
    return  - (metrics['Max Memory Used'] / 1024) - new_timeout-cost

# Function to adjust configuration based on performance
def adjust_configuration_based_on_performance(metrics, memory, timeout):
    if metrics['out_of_memory']:
        memory = min(memory + 128, 3008)  # Increase memory by 128MB
    if metrics['timed_out']:
        timeout = min(timeout + 1, 15)  # Increase timeout by 1 second
    return memory, timeout

# Function to choose action using epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    state = np.array([state])
    act_values = model.predict(state)
    return np.argmax(act_values[0])

# Function to execute action
def execute_action(memory, timeout, action):
    if action == 0 and memory < 3008:  # Increase memory
        memory += 1
    elif action == 1 and memory > 128:  # Decrease memory
        memory -= 1
    elif action == 2 and timeout < 15:  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > 1:  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    return memory, timeout

# Function to replay experiences from the buffer
def replay(batch_size):
    minibatch = random.sample(memory_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(target_model.predict(np.array([next_state]))[0])
        target_f = model.predict(np.array([state]))
        target_f[0][action] = target
        model.fit(np.array([state]), target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Training Loop
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
logs_client = boto3.client('logs')
log_group_name = '/aws/lambda/x22203389-ric-resize'
func_name='x22203389-ric-resize'

bucket_name = 'x22203389-ric'
folder_path = '51000/'
s3_objects = helper.get_image_list_from_s3(s3_client,bucket_name,folder_path)

for episode in range(num_episodes):
    try:
        memory, timeout = 512, 5  # Initial configuration
        state = [memory, timeout, 0]
        
        # Get a random image from S3 bucket
        object_key = random.choice(s3_objects)
        print(f"Episode number: {episode}, Image: {object_key}")
        
        # Define the payload for the Lambda function
        payload = {
            'bucket_name': bucket_name,
            'object_key': object_key
        }
        
        for step in range(max_steps_per_episode):
            action = choose_action(state, epsilon)
            new_memory, new_timeout = execute_action(memory, timeout, action)
            
            # Update Lambda function configuration
            lambda_client.update_function_configuration(
                FunctionName=func_name,
                MemorySize=new_memory,
                Timeout=new_timeout
            )
            
            # Wait for the configuration to apply
            time.sleep(5)  # Adjust the sleep time as needed
            
            # Invoke the Lambda function with the updated configuration
            response = lambda_client.invoke(
                FunctionName=func_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload),
                Qualifier='$LATEST'
            )
            
            # Get the response and extract metrics
           # response_payload = json.loads(response['Payload'].read())
            
            # Wait for logs to propagate to CloudWatch
            time.sleep(60)  # Adjust the sleep time as needed
            
            # Fetch and parse the latest CloudWatch log entry
            end_time = datetime.utcnow()+timedelta(minutes=1)
            start_time = end_time - timedelta(minutes=5)
            log_event = helper.get_latest_log_stream(logs_client,log_group_name,2)
            if log_event:
                metrics = helper.parse_log_event(log_event)
                print(metrics)
                duration = metrics['Duration']
                max_memory_used = metrics['Max Memory Used']
                
                duration_category = get_duration_category(duration)
                next_state = [new_memory, new_timeout, duration_category]
                
                reward = calculate_reward(metrics, new_memory, new_timeout)
                
                memory, timeout = adjust_configuration_based_on_performance(metrics, new_memory, new_timeout)
                
                done = step == max_steps_per_episode - 1
                memory_buffer.append((state, action, reward, next_state, done))
                
                state = next_state
            
            if len(memory_buffer) > batch_size:
                replay(batch_size)
        
        print(f'Episode {episode + 1}/{num_episodes} completed.')
        
        # Update the target model periodically
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
            model.save('dqn_model.h5')
    except Exception as e:
        print(f'Exception occurred: {e}')
        model.save('dqn_model.h5')
        break  # Optionally break the loop or continue

# Save the final model
model.save('dqn_model.h5')
