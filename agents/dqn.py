import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import losses

import random
from collections import deque
import boto3
import json
import time
import os
from datetime import datetime, timedelta
import helper
# Define the environment and DQN parameters
memory_sizes = np.arange(128, 3009)  # Continuous memory sizes from 128MB to 3008MB
timeouts = [1, 2, 3, 5, 10, 15]      # Discrete timeout values
duration_categories = [i for i in range(0, 150001, 500)] # Updated duration categories

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
num_episodes = 100
max_steps_per_episode = 10

model_filename = 'dqn_new_model.h5'
results_file='dqn_results.json'
q_table_file_path = 'dqn_table.txt'
target_model_file= 'dqn_new_model.h5'

log_group_name ='/aws/lambda/x22203389-ric-rotation'
func_name='x22203389-ric-rotation'
bucket_name = 'x22203389-ric'
folder_path = '51000/'

#bucket_name = 'x22203389-imageset'
# Function to save the model
def save_model(model, filename):
    model.save(filename)

# Function to load the model
def load_model(filename):
    if os.path.exists(model_filename):
        mod=models.load_model(filename)
        mod.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())
        return mod
       # target_model = load_model(model_filename)
        #target_model.set_weights(model.get_weights())
    else:
        return build_model(state_size, action_size)
       # target_model = build_model(state_size, action_size)
        #target_model.set_weights(model.get_weights())
    #return model

def load_target_model(model,filename):
    if os.path.exists(model_filename):
       # model = models.load_model(filename)
        target_model = models.load_model(filename)
        target_model.set_weights(model.get_weights())
        return target_model
    else:
        #model = build_model(state_size, action_size)
        target_model = build_model(state_size, action_size)
        target_model.set_weights(model.get_weights())
        return target_model

# Define the DQN model
def build_model(state_size, action_size):
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

model = load_model(model_filename)
target_model = load_target_model(model,target_model_file)


# Function to get the duration category
def get_duration_category(duration):
    for i, threshold in enumerate(duration_categories):
        if duration <= threshold:
            return i
    return len(duration_categories) - 1


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
    return  - (metrics['Max Memory Used'] / 1024) -cost

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
        memory += 16
    elif action == 1 and memory > 128:  # Decrease memory
        memory -= 16
    elif action == 2 and timeout < 15:  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > 1:  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    return memory, timeout

# Function to replay experiences from the buffer
def replay(batch_size):
    global epsilon
    global model
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

def get_random_initial_configuration():
    memory, timeout = 256, 5 
    if np.random.rand() < epsilon:
        memory = np.random.choice(memory_sizes)
        timeout = np.random.choice(timeouts)
    return memory, timeout


# Training Loop
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
logs_client = boto3.client('logs')

s3_objects = helper.get_image_list_from_s3(s3_client,bucket_name,folder_path)


results = []
for episode in range(11,num_episodes):
    try:
        memory, timeout = get_random_initial_configuration()  # Initial configuration
        state = [memory, timeout, 0]
        
        # Get a random image from S3 bucket
        object_key = random.choice(s3_objects)
        print(f"Episode number: {episode}, Image: {object_key}")
        
        # Define the payload for the Lambda function
        payload = {
            'bucket_name': bucket_name,
            'object_key': object_key
        }
        res= {
            'episodes': episode,
            'execution_times': [],
            'memory_usages': [],
            'costs': [],
            'rewards': [],
            'memory_configurations': [],
            'timeout_configurations': []
        }
        for step in range(max_steps_per_episode):
            action = choose_action(state, epsilon)
            print(f'Action {action}')
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
                res['execution_times'].append(duration)
                res['memory_usages'].append(max_memory_used)
                res['costs'].append(calculate_cost(memory, duration))
                res['rewards'].append(reward)
                res['memory_configurations'].append(memory)
                res['timeout_configurations'].append(timeout)
            if len(memory_buffer) > batch_size:
                replay(batch_size)
            
            results.append(res)
       
        
        # Save the Q-table to a file
        # Save the Q-table to a text file
        with open(q_table_file_path, 'w') as f:
            for experience in memory_buffer:
                state, action, reward, next_state, done = experience
                f.write(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}\n")

        helper.append_results_to_file(res,results_file)
        print(f'Episode {episode + 1}/{num_episodes} completed.')
        
        # Update the target model periodically
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
            model.save(model_filename,include_optimizer=False)
            target_model.save('target_model.h5',include_optimizer=False)
    except Exception as e:
        print(f'Exception occurred: {e}')
        with open(q_table_file_path, 'w') as f:
            for experience in memory_buffer:
                state, action, reward, next_state, done = experience
                f.write(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}\n")


        model.save(model_filename,include_optimizer=False)
        target_model.save('target_model.h5',include_optimizer=False)
        # Save the results to a file
        with open('results.json', 'w') as f:
            json.dump(results, f)
        break  # Optionally break the loop or continue

# Save the final model
model.save(model_filename,include_optimizer=False)
target_model.save('target_model.h5')
# Save the results to a file
with open('results.json', 'w') as f:
    json.dump(results, f)
