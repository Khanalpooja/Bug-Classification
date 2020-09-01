import boto3
from github import Github, Issue
import os 
access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
print(access_key_id)
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
print(secret_access_key)
client = boto3.resource('dynamodb', endpoint_url='http://localhost:8000', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
# client = boto3.client('dynamodb')

def repo_details(repo_name):
    # First create a Github instance: using an access token
    g = Github("1954b8f76b456ec6b1c37c55504f8168fe08c81d")
    # repo_name = event["repo_name"]
    print(repo_name)
    repo = g.get_repo(repo_name)
    repo_id = repo.id
    repo_full_name = repo.full_name
    description = repo.description
    languages = repo.get_languages() 
    open_issues_count = repo.open_issues_count
    size= repo.size
    created_date_time= repo.created_at
    updated_date_time = repo.updated_at
    organization= repo.organization
    contributors = repo.get_contributors()
    branches= repo.get_branches()
    commits= repo.get_commits()
    deployments = repo.get_deployments()
    projects = repo.get_projects()
    releases = repo.get_releases()

    contributor_count = 0
    branch_count = 0
    commit_count = 0
    deployment_count = 0
    project_count = 0
    release_count = 0
    lang_dict = dict()


    for key in languages:
        lang_dict.update({key : {'N' : str(languages[key])}})

    
    for contributor in contributors:
        contributor_count +=1
        
    for branch in branches:
        branch_count +=1
          
    for release in releases:
        release_count +=1
        
    for commit in commits:
        commit_count +=1
    
    for deployment in deployments:
        deployment_count +=1
        
    for project in projects:
        project_count +=1
    table = client.Table('Movies')
    response = table.put_item(
        # TableName='repo_details',
        Item={
            'repo_id': {
                'S': str(repo_id)
                },
            'repo_name' : {
                'S': repo_full_name
                },
            'Description' : {
                'S' : description
                },
            'Language': {
                'M' : lang_dict
                },
            'Size': {
                'N': str(size)
                },
            'Created_date_time' : {
                'S' : str(created_date_time)
                },
            'Updated_date_time' : {
                'S' : str(updated_date_time)
                },
            'Organization' : {
                'S' : str(organization)
                },
            'Contributors' : {
                'N' : str(contributor_count)
                }, 
            'Branches' : {
                'N' : str(branch_count)
                },
            'Commits' : {
                'N' : str(commit_count)
                },
            'Deployments' : {
                'N' : str(deployment_count)
                },
            'Projects' : {
                'N' : str(project_count)
                },
            'Releases' : {
                'N': str(release_count)
                }    
            }
    )

def main():
    print("this is main")  
    repo_details("mojombo/grit")


if __name__ == '__main__':
   main()
    
    
    