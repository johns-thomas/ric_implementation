import boto3
import time

from botocore.config import Config

my_config = Config(
    region_name = 'eu-west-1',
    
)
# Initialize boto3 clients
lambda_client = boto3.client('lambda',config=my_config)
cloudwatch_client = boto3.client('cloudwatch',config=my_config)


def invoke_lambda(memory, timeout, concurrency, function_name):
    lambda_client.update_function_configuration(
        FunctionName=function_name,
        MemorySize=memory,
        Timeout=timeout,
        #ReservedConcurrentExecutions=concurrency
    )
    
    time.sleep(5)  # Wait for the configuration update
    
    invoke_response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse'
    )
    
    time.sleep(30)  # Simulate delay for CloudWatch metrics
    
    return invoke_response['LogResult']


def parse_log_event(log_event_str, log_stream_str):
    log_event = log_event_str['value']
    log_stream = log_stream_str['value']

    # Initialize the metrics dictionary
    metrics = {'out_of_memory': False, 'timed_out': False}
    
    # Split the log event string by tabs
    parts = log_event.split('\t')

    #print(parts)
    for part in parts:
        val=part.split(':')
        if val[0]=='Duration':
            metrics['Duration'] = float(val[1].replace(' ms', ''))
        elif val[0]=='Billed Duration':
            metrics['Billed Duration'] = float(val[1].replace(' ms', ''))
        elif val[0]=='Memory Size':
            metrics['Memory Size'] = int(val[1].replace(' MB', ''))
        elif val[0]=='Max Memory Used':
            metrics['Max Memory Used'] = int(val[1].replace(' MB', ''))
        elif val[0]=='Init Duration' :
            metrics['Init Duration'] = float(val[1].replace(' ms', ''))
        if 'Task timed out' in part:
            metrics['timed_out'] = True
        if 'fatal error' in part and 'Cannot allocate memory' in part:
            metrics['out_of_memory'] = True

    # Extracting additional information from the log stream
    log_stream_parts = log_stream.split('/')
    metrics['date'] = log_stream_parts[0]
    metrics['version'] = log_stream_parts[2]

    return metrics

# Example usage
log_event_str = {'field': '@message', 'value': 'REPORT RequestId: 882e426b-1543-45c0-9671-65992197623f\tDuration: 1897.26 ms\tBilled Duration: 1898 ms\tMemory Size: 512 MB\tMax Memory Used: 121 MB\tInit Duration: 343.31 ms\t\n'}
log_stream_str = {'field': '@logStream', 'value': '2024/07/16/[$LATEST]00e03ae5c9fb410291af735ccab9cde4'}

metrics = parse_log_event(log_event_str, log_stream_str)
print(metrics)
