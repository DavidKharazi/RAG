import boto3
import os

session = boto3.session.Session()
s3_client = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id='YCAJEt7ilkMDiPuuZA--Sgb1H',
    aws_secret_access_key='YCOJE46MLMRlPll_kl6oIllqvT7P7S65E4QohXLZ'
)
def upload_file_to_bucket(file_name, bucket, local_directory, bucket_directory, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param local_directory: Local directory where the file is located
    :param bucket_directory: Directory within the bucket to upload the file to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    if object_name is None:
        object_name = os.path.basename(file_name)

    local_path = os.path.join(local_directory, file_name)
    s3_path = os.path.join(bucket_directory, object_name)

    try:
        response = s3_client.upload_file(local_path, bucket, s3_path)
    except Exception as e:
        print(e)
        return False
    return True
upload_file_to_bucket('image_4.png', 'utlik', r'C:\Users\datoh\PycharmProjects\RAG_pro\output_images', 'RELAXSAN/images/', 'image_4.png')
