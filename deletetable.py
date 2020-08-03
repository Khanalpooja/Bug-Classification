
import boto3

dynamodb = boto3.resource('dynamodb', 'us-east-1')
table = dynamodb.Table('issue_details')

scan = table.scan(
    ProjectionExpression='#k',
    ExpressionAttributeNames={
        '#k': 'name'
    }
)

with table.batch_writer() as batch:
    for each in scan['Items']:
        batch.delete_item(Key=each)
