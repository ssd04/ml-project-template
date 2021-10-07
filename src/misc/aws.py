import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from loguru import logger


aws_region = os.environ.get("AWS_REGION")
config = Config(retries={"max_attempts": 5, "mode": "adaptive"})


def get_ssm_parameter_value(parameter_name):
    try:
        ssm = boto3.client("ssm", region_name=aws_region, config=config)
        parameter = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
    except client.exceptions.ParameterNotFound:
        logger.exception(f"Parameter [{parameter_name}] not found")
    else:
        return parameter["Parameter"]["Value"]


def read_data_from_s3(bucket_name, object_key):
    """
    Gets an object from a bucket.

    :param bucket: The bucket that contains the object.
    :param object_key: The key of the object to retrieve.
    :return: The object data in bytes.
    """

    try:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)
        body = bucket.Object(object_key).get()["Body"].read()
        post_data = body

        if isinstance(body, (bytes, bytearray)):
            try:
                post_data = body
            except IOError:
                logger.exception("Expected file name or binary data, got '%s'.", body)
                raise

        logger.info(f"Got object '{object_key}' from bucket '{bucket.name}'.")
    except ClientError:
        logger.exception(
            ("Couldn't get object '%s' from bucket '%s'.", object_key, bucket.name)
        )
        raise
    else:
        return post_data


def save_data_to_s3(bucket_name, object_key, data):
    """
    Upload data to a bucket and identify it with the specified object key.

    :param bucket: The bucket to receive the data.
    :param object_key: The key of the object in the bucket.
    :param data: The data to upload. This can either be bytes or a string. When this
                 argument is a string, it is interpreted as a file name, which is
                 opened in read bytes mode.
    """
    put_data = data
    if isinstance(data, str):
        try:
            put_data = bytearray(data.encode("ascii"))
        except IOError:
            logger.exception("Expected file name or binary data, got '%s'.", data)
            raise

    try:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)

        obj = bucket.Object(object_key)
        obj.put(Body=put_data)
        obj.wait_until_exists()
        logger.info(f"Put object '{object_key}' to bucket '{bucket.name}'.")
    except ClientError:
        logger.exception(
            "Couldn't put object '%s' to bucket '%s'.", object_key, bucket.name
        )
        raise
    finally:
        if getattr(put_data, "close", None):
            put_data.close()


def get_sns_topic_by_name(client, topic_name):
    try:
        response = client.list_topics()
    except ClientError as e:
        logger.error(f"Failed with error: {e}")
        raise

    for topic in response["Topics"]:
        topic_name_tmp = topic["TopicArn"].split(":")[-1]
        if topic_name_tmp == topic_name:
            return topic["TopicArn"]
    return


def send_notification(topic_name, message, subject):
    try:
        client = boto3.client("sns", region_name=aws_region, config=config)

        topic_arn = get_sns_topic_by_name(client, topic_name)

        if topic_arn is not None:
            client.publish(
                TopicArn=topic_arn,
                Message=message,
                Subject=subject,
            )
    except ClientError as e:
        logger.error(f"Failed with error: {e}")
        raise
    return
