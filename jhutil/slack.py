import requests
import json
from .secret import channel_to_webhook_url
from slack_sdk import webhook, WebClient
import random
from datetime import datetime
import jhutil


def send_slack(message, channel="gpu"):
    webhook_url = channel_to_webhook_url[channel]
    
    payload = {
        "text": "<!channel>",
        "attachments": [{"text": message}]
    }
    headers = {
        "Content-type": "application/json; charset=utf-8"
    }
    response = requests.post(
        webhook_url, data=json.dumps(payload), headers=headers)
    
    # check response is valid
    if response.status_code != 200:
        # warning message
        print("Slack webhook failed with status code: {}".format(
            response.status_code))
        return False
    return True