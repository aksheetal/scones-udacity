# serializeImageData:
import json
import boto3
import base64
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

s3 = boto3.resource('s3')
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
logger.debug("Using SageMaker endpoint: %s", str(ENDPOINT_NAME))
THRESHOLD = os.environ["THRESHOLD"]

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    try:
        s3.Bucket(bucket).download_file(key, '/tmp/image.png')
    except Exception as e:
        print("Something went wrong with downloading obj to tmp folder")
        raise e

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



###################################################################
#inferenceSerializer

import json
import boto3
import base64

sagemaker_client = boto3.client('runtime.sagemaker')



def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint
    response = sagemaker_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/x-image',
        Body=image
    )
    inferences = response["Body"].read().decode('utf-8')

    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')]

    return {
        'statusCode': event["statusCode"],
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": event["inferences"]
        }
    }


#################################################################
#filterConfidences

# we need to filter low-confidence inferences when greather than given threshold

import json

class Threshold_Error(Exception):
    pass

def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event["body"]["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(i > THRESHOLD for i in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Threshold_Error("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event["body"]
    }
