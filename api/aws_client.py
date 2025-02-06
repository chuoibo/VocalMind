import uuid
import boto3
import logging

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from utils.api_logger import logging
from config.app_config import Config as cfg


class AWSS3Bucket:
    def __init__(self):
        self.aws_access_key_id = cfg.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = cfg.AWS_SECRET_ACCESS_KEY
        self.region_name = cfg.AWS_REGION
        self.bucket_name = cfg.BUCKET_NAME
        logging.info('Initialized aws s3 bucket ...')


    def get_s3_client(self):
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )
            logging.info("Credentials loaded successfully!")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logging.info("Failed to load credentials:", e)
        
        return s3_client


    def upload_to_s3(self, file, filename):
        logging.info('Start to upload file into S3 bucket')
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        s3_client = self.get_s3_client()

        s3_client.upload_fileobj(
            file,
            self.bucket_name,
            unique_filename,
            ExtraArgs={"ACL": "private"}  
        )

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': unique_filename},
            ExpiresIn=3600 
        )

        logging.info(f'Finish uploading file into S3 bucket with presigned_url: {presigned_url}')

        return presigned_url