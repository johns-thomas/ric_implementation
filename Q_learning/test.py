import boto3
import random
import os
import json
import time
import datetime




# Initialize a session using Amazon S3
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

def get_latest_log_stream(logs_client, log_group_name,request_id):
    response = logs_client.describe_log_streams(
        logGroupName=log_group_name,
        orderBy='LastEventTime',
        descending=True,
        limit=1
    )
    if 'logStreams' in response and len(response['logStreams']) > 0:
        latest_log_stream_name =  response['logStreams'][0]['logStreamName']
        
# Get the latest log stream
        #get_latest_log_stream(logs_client, log_group_name)
        if latest_log_stream_name:
            print(f"Latest log stream: {latest_log_stream_name}")
    # Query the latest log stream
            query_log_stream(logs_client, log_group_name, latest_log_stream_name)
        else:
            print("No log streams found.")
    
    return None

def query_log_stream(logs_client, log_group_name, log_stream_name):
    query_string =f"""
  fields @timestamp, @message, @logStream
    | filter @logStream = "{log_stream_name}" and  @message like /Max Memory Used/ 
    | sort @timestamp desc
    | limit 1
    """
    
    start_query_response = logs_client.start_query(
        logGroupName=log_group_name,
       #logStreamNames=[log_stream_name,],
        startTime=int((datetime.now() - timedelta(minutes=2)).timestamp()),
        endTime=int((datetime.now() + timedelta(seconds=30)).timestamp()),
        queryString=query_string,
    )
    
    query_id = start_query_response['queryId']
    # Wait for the query to complete
    
    while True:
        query_status = logs_client.get_query_results(queryId=query_id)
        print(query_status)
        if query_status['status'] == 'Complete':
            break
        time.sleep(1)

# Print the results
    for result in query_status['results']:
        print(result)



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

# Usage example:
bucket_name = 'x22203389-ric'
folder_path = '51000/'
image_files = get_image_list_from_s3(bucket_name,folder_path)
random_image =  random.choice(image_files)#
#object_key = random_image['Key']
print(random_image)

payload = {
        'bucket_name': bucket_name,
        'object_key': random_image
    }
response = lambda_client.invoke(
            FunctionName='x22203389-ric-resize',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
            Qualifier='$LATEST'
        )
response_payload = response['Payload'].read().decode('utf-8')
request_id = response['ResponseMetadata']['RequestId']
time.sleep(10)  




logs_client = boto3.client('logs')

log_group_name = '/aws/lambda/x22203389-ric-resize'

# Query the CloudWatch Logs using the RequestId
max_memory_used = get_latest_log_stream(logs_client, log_group_name, request_id)
print(f"Max memory used during the Lambda invocation: {max_memory_used} MB")
