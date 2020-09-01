#crawler to get data from github and store in amazon dynamodb

from github import Github, Issue
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime
import calendar
import time 



# creating github object using an access token
g = Github(<session_id>)


def handle_rate_limit():
        print("Handling rate limit")
        core_rate_limit = g.get_rate_limit().core
        remaining_limit = core_rate_limit.remaining
        print("remaining limit", remaining_limit)
        if remaining_limit <= 100:
                reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                print("reset_timestamp", reset_timestamp)
                sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 5  # add 5 seconds to be sure the rate limit has been reset
                print("sleep time", sleep_time)
                time.sleep(sleep_time)

with ThreadPoolExecutor(max_workers=100) as executor:
        i = 0
        handle_rate_limit()
        for repo in g.get_repos(0):
                handle_rate_limit()
                # if i ==500 :
                #         break
                print("number of repo processed", i)
                repo_full_name = repo.full_name
                print(repo_full_name)
                executor.submit(requests.get,
                f"https://rif93n1z2m.execute-api.us-east-1.amazonaws.com/dev/get-repo-details/?repo_name={repo_full_name}")
                i +=1



