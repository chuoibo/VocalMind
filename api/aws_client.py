from dotenv import load_dotenv
import uuid
import os
import boto3

load_dotenv()

BUCKET_NAME = "speech-ai-files"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

def upload_to_s3(file, filename):
    unique_filename = f"{uuid.uuid4()}_{filename}"
    
    # Upload file to S3
    s3_client.upload_fileobj(
        file,
        BUCKET_NAME,
        unique_filename,
        ExtraArgs={"ACL": "private"}  
    )
    
    file_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{unique_filename}"
    return file_url