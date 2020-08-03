from github import Github, Issue
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# First create a Github instance:

# using username and password
# g = Github("user", "password")

# or using an access token
g = Github("366fe227c7a9daf4e6cea243a1a824f31b0a40d3")
executor = ThreadPoolExecutor(max_workers=3)

def parallel():
        i = 0
        for repo in g.get_repos():
                if i == 2:
                        break
                #     print(repo)
                print(repo.full_name)
                res = requests.get(f"https://rif93n1z2m.execute-api.us-east-1.amazonaws.com/dev/get-repo-details/?repo_name={repo.full_name}")
                #     print(res.content)
                i += 1


task1 = executor.submit(parallel)
task2 = executor.submit(parallel)
task3 = executor.submit(parallel)


