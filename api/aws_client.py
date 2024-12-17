import uuid
import os
import boto3
import logging

from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

from utils.api_logger import logging


load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
logging.info(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")

aws_secret_access_key =os.getenv("AWS_SECRET_ACCESS_KEY")
logging.info(f"AWS_ACCESS_KEY_ID: {aws_secret_access_key}")

region_name = os.getenv("AWS_REGION")
logging.info(f"AWS_ACCESS_KEY_ID: {region_name}")


BUCKET_NAME = "speechfsds"

try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    logging.info("Credentials loaded successfully!")
except (NoCredentialsError, PartialCredentialsError) as e:
    logging.info("Failed to load credentials:", e)


def upload_to_s3(file, filename):
    logging.info('Start to upload file into S3 bucket')
    unique_filename = f"{uuid.uuid4()}_{filename}"
    
    # Upload file to S3
    s3_client.upload_fileobj(
        file,
        BUCKET_NAME,
        unique_filename,
        ExtraArgs={"ACL": "private"}  
    )

    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': unique_filename},
        ExpiresIn=3600 
    )

    logging.info(f'Finish uploading file into S3 bucket with presigned_url: {presigned_url}')

    return presigned_url