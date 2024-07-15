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


invoke_lambda(128,3,2,'x22203389-resize')