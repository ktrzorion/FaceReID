import boto3
from botocore.exceptions import ClientError
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class S3Helper:

    s3_access_key = os.environ['S3_ACCESS_KEY']
    s3_secret_key = os.environ['S3_SECRET_KEY']
    s3_client = ""
    s3_bucket = os.environ['S3_BUCKET']

    def __init__(self):
        config = Config(
        region_name='ap-south-1',
        signature_version='s3v4'
        )
        print(self.s3_access_key, self.s3_secret_key)
        self.s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
            config=config
        )

    def create_presigned_url(self, object_name: str, expiration=10800, bucket_name= s3_bucket):
        """Generate a presigned URL to share an S3 object

        :param bucket_name: string
        :param object_name: string
        :param expiration: Time in seconds for the presigned URL to remain valid
        :return: Presigned URL as string. If error, returns None."""
        # Generate a presigned URL for the S3 object
        try:
            response = self.s3_client.generate_presigned_url('get_object',
                                                        Params={'Bucket': bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=expiration)
        except ClientError as e:
            return None
        except Exception as e:
            return None
        # The response contains the presigned URL
        return response

    def upload_file(self, file_path: str, object_name: Optional[str] = None) -> bool:
        """
        Upload a file to S3 with enhanced validation and verification
        
        Args:
            file_path: Path to local file
            object_name: Optional custom S3 object name
            
        Returns:
            bool: True if upload successful and verified
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                raise ValueError(f"Invalid file path: {file_path}")
                
            object_name = object_name or file_path.name
            file_size = file_path.stat().st_size
            
            logger.info(f"Uploading file: {file_path} ({file_size} bytes) to s3://{self.s3_bucket}/{object_name}")

            # Upload with progress callback
            self.s3_client.upload_file(
                str(file_path), 
                self.s3_bucket, 
                object_name,
                Callback=lambda bytes_transferred: logger.debug(
                    f"Upload progress: {bytes_transferred}/{file_size} bytes"
                )
            )
            
            # Verify upload
            if self.verify_file_exists(object_name):
                logger.info(f"Successfully uploaded and verified: {object_name}")
                return True
            
            logger.error("Upload verification failed")
            return False
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            return False

    def verify_file_exists(self, object_name: str) -> bool:
        """
        Verify if a file exists in the S3 bucket

        Args:
            object_name (str): S3 object name to verify

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=object_name)
            return True
        except ClientError:
            return False

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from S3

        Args:
            object_name (str): S3 object name to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=object_name)
            logging.info(f"Successfully deleted {object_name} from {self.s3_bucket}")
            return True
        except ClientError as e:
            logging.error(f"Failed to delete file from S3: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error deleting file: {str(e)}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """
        List files in S3 bucket with optional prefix

        Args:
            prefix (str): Prefix to filter objects (optional)

        Returns:
            list: List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)
            files = [obj['Key'] for obj in response.get('Contents', [])]
            return files
        except ClientError as e:
            logging.error(f"Failed to list files in S3: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error listing files: {str(e)}")
            return []