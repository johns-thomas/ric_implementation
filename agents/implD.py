import numpy as np
import boto3
import random
import time
import json
from datetime import datetime, timedelta
import helper 


# Define the ranges and categories
memory_sizes = [i for i in range(128, 3009, 128)]
timeouts = [1, 2, 3, 5, 10, 15]
duration_categories = [i for i in range(0, 150001, 500)]  # Define duration categories in milliseconds

Q_shape = (len(memory_sizes) * len(timeouts) * len(duration_categories), 6)
q_table_file_path = 'q_table-new.npy'
state_data_path='qstate-new.txt'
results_file='results_q.json'

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

log_group_name ='/aws/lambda/x22203389-ric-rotation'
func_name='x22203389-ric-rotation'

Q = helper.load_q_table(q_table_file_path,Q_shape)
#print(Q)
def get_duration_category(duration):
    for i, threshold in enumerate(duration_categories):
        if duration <= threshold:
            return i
    return len(duration_categories) - 1

def get_state_index(memory, timeout, duration_category):
    memory_index = memory_sizes.index(memory)
    timeout_index = timeouts.index(timeout)
    return (memory_index * len(timeouts) * len(duration_categories) +
            timeout_index * len(duration_categories) +
            duration_category)

def choose_action(state, Q):
    print(np.argmax(Q[state]))
    if np.random.rand() < epsilon:
        return np.random.randint(6)
    return np.argmax(Q[state])

def execute_action(memory, timeout, action):
    min_memory = 128  # Minimum memory to avoid failures
    if action == 0 and memory < memory_sizes[-1]:  # Increase memory
        memory = memory_sizes[memory_sizes.index(memory) + 1]
    elif action == 1 and memory > min_memory:  # Decrease memory
        memory = memory_sizes[max(0, memory_sizes.index(memory) - 1)]
    elif action == 2 and timeout < timeouts[-1]:  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > timeouts[0]:  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    return memory, timeout

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

def calculate_cost(memory, duration):
    memory = int(memory) 
    duration = float(duration) 
    return (memory / 1024) * duration * 0.00001667

def calculate_reward(metrics, new_memory, new_timeout):
    if metrics['out_of_memory']:
        return -100  # Large penalty for out-of-memory errors
    if metrics['timed_out']:
        return -50  # Penalty for timeouts
    # Reward based on performance and resource usage
    cost = calculate_cost(new_memory, metrics['Duration'])
    return  - (metrics['Max Memory Used'] / 1024) - new_timeout-cost

def adjust_configuration_based_on_performance(metrics, memory, timeout):
    if metrics['out_of_memory']:
        memory = min(memory * 2, memory_sizes[-1])  # Double memory
    if metrics['timed_out']:
        timeout = min(timeout * 2, timeouts[-1])  # Double timeout
    return memory, timeout


# Training Loop
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
cloudwatch_client = boto3.client('cloudwatch')
logs_client = boto3.client('logs')

bucket_name = 'x22203389-ric'
folder_path = '51000/'
s3_objects = helper.get_image_list_from_s3(s3_client,bucket_name,folder_path)
# Initialize the results dictionary
results = []
for episode in range(78,num_episodes):
    try:
        memory, timeout = 512, 5  # Initial configuration
        state = get_state_index(memory, timeout, 0)
        
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

        for step in range(10):
            action = choose_action(state, Q)
            new_memory, new_timeout = execute_action(memory, timeout, action)
            print(f'Action {action}')
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
            #response_payload = json.loads(response['Payload'].read())
            print(response)
            # Wait for logs to propagate to CloudWatch
            time.sleep(60)  # Adjust the sleep time as needed
            
            # Fetch and parse the latest CloudWatch log entry
            end_time = datetime.utcnow()+timedelta(minutes=1)
            start_time = end_time - timedelta(minutes=5)
            
            log_event = helper.get_latest_log_stream(logs_client,log_group_name,2)
            if log_event:
                metrics = helper.parse_log_event(log_event)
                duration = metrics['Duration']
                max_memory_used = metrics['Max Memory Used']
                print(metrics)
                duration_category = get_duration_category(duration)
                new_state = get_state_index(new_memory, new_timeout, duration_category)
                
                reward = calculate_reward(metrics, new_memory, new_timeout)
                
                Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
                
                memory, timeout = adjust_configuration_based_on_performance(metrics, new_memory, new_timeout)
                state = new_state
                res['execution_times'].append(duration)
                res['memory_usages'].append(max_memory_used)
                res['costs'].append(calculate_cost(memory, duration))
                res['rewards'].append(reward)
                res['memory_configurations'].append(memory)
                res['timeout_configurations'].append(timeout)
         
        results.append(res)

        print(f'Episode {episode + 1}/{num_episodes} completed.')
        #print(f'Reward {total_reward}')
    
        helper.append_results_to_file(res,'resnew.json')


        print(f'Episode {episode + 1}/{num_episodes} completed.')
        helper.save_q_table(q_table_file_path, Q)
    
    except Exception as e:
        print(f'Exception occurred: {e}')
        helper.append_results_to_file(results,results_file)

        helper.save_q_table(q_table_file_path, Q)
        break  

helper.append_results_to_file(results,results_file)

helper.save_q_table(q_table_file_path, Q)
