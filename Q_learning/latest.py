import numpy as np
import boto3
from datetime import datetime, timedelta
import random
import time
import json
import random
import os

# Define the ranges and categories
memory_sizes = [128, 256, 512, 1024, 2048, 3008]
timeouts = [1, 2, 3, 5, 10, 15]
cpu_utilization_levels = [0,10,25,40, 50, 80, 100]
#image_size_categories = [0, 500000, 1000000, 5000000]

Q_shape = (len(memory_sizes) * len(timeouts) * len(cpu_utilization_levels) )
Q = np.zeros(Q_shape)

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

#Helper methods

def get_image_list_from_s3(bucket_name, folder_path):
    # List objects within the specified folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    # Check if 'Contents' in response
    if 'Contents' not in response:
        print(f"No objects found in the bucket {bucket_name} with prefix {folder_path}")
        return None

    # Filter to include only image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [
        obj['Key'] for obj in response['Contents']
        if os.path.splitext(obj['Key'])[1].lower() in image_extensions
    ]

    # Check if any image files are found
    if not image_files:
        print(f"No image files found in the bucket {bucket_name} with prefix {folder_path}")
        return None

   
    
    return image_files



def get_image_size_category(image_size):
    for i, threshold in enumerate(image_size_categories):
        if image_size <= threshold:
            return i
    return len(image_size_categories) - 1

def get_cpu_utilization_category(cpu_utilization):
    for i, threshold in enumerate(cpu_utilization_levels):
        if cpu_utilization <= threshold:
            return i
    return len(cpu_utilization_levels) - 1

def get_state_index(memory, timeout, cpu_utilization, image_size_category):
    memory_index = memory_sizes.index(memory)
    timeout_index = timeouts.index(timeout)
    cpu_utilization_index = get_cpu_utilization_category(cpu_utilization)
    return (memory_index * len(timeouts) * len(cpu_utilization_levels) * len(image_size_categories) +
            timeout_index * len(cpu_utilization_levels) * len(image_size_categories) +
            cpu_utilization_index * len(image_size_categories) +
            image_size_category)

def choose_action(state, Q):
    if np.random.rand() < epsilon:
        return np.random.randint(6)
    return np.argmax(Q[state])

def execute_action(memory, timeout, action):
    if action == 0 and memory < memory_sizes[-1]:  # Increase memory
        memory = memory_sizes[memory_sizes.index(memory) + 1]
    elif action == 1 and memory > memory_sizes[0]:  # Decrease memory
        memory = memory_sizes[memory_sizes.index(memory) - 1]
    elif action == 2 and timeout < timeouts[-1]:  # Increase timeout
        timeout = timeouts[timeouts.index(timeout) + 1]
    elif action == 3 and timeout > timeouts[0]:  # Decrease timeout
        timeout = timeouts[timeouts.index(timeout) - 1]
    return memory, timeout

# Training Loop
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
cloudwatch_client = boto3.client('cloudwatch')
bucket_name = 'x22203389-ric'
folder_path = '51000/'
image_files = get_image_list_from_s3(bucket_name,folder_path)

for episode in range(num_episodes):
    memory, timeout = 512, 5  # Initial configuration
    state = get_state_index(memory, timeout, 0, 0)
    
    # Get a random image from S3 bucket
   
    random_image =  random.choice(image_files)
    #object_key = random_image['Key']
    
    # Define the payload for the Lambda function
    payload = {
        'bucket_name': bucket_name,
        'object_key': random_image
    }
    
    for step in range(10):
        action = choose_action(state, Q)
        new_memory, new_timeout = execute_action(memory, timeout, action)
        
        # Update Lambda function configuration
        lambda_client.update_function_configuration(
            FunctionName='your_lambda_function_name',
            MemorySize=new_memory,
            Timeout=new_timeout
        )
        
        # Wait for the configuration to apply
        time.sleep(5)  # Adjust the sleep time as needed
        
        # Invoke the Lambda function with the updated configuration
        response = lambda_client.invoke(
            FunctionName='your_lambda_function_name',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
            Qualifier='$LATEST'
        )
        
        # Get the response and extract metrics
        response_payload = json.loads(response['Payload'].read())
        metrics = json.loads(response_payload['body'])
        
        duration = metrics['Duration']
        cpu_utilization = get_current_cpu_utilization(cloudwatch_client, 'your_lambda_function_name')
        image_size = metrics['ImageSize']
        
        image_size_category = get_image_size_category(image_size)
        cpu_utilization_category = get_cpu_utilization_category(cpu_utilization)
        
        new_state = get_state_index(new_memory, new_timeout, cpu_utilization, image_size_category)
        
        # Dummy reward calculation
        reward = 100 - (new_memory / 1024) - new_timeout
        
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
        
        memory, timeout = new_memory, new_timeout
        state = new_state
    
    print(f'Episode {episode + 1}/{num_episodes} completed.')

# Save Q-table
np.save('q_table.npy', Q)






