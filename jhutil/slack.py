import requests
import json

def send_slack(message, channel="gpu"):
    if channel == "gpu":
        webhook_url = "https://hooks.slack.com/services/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"
    elif channel == "jinhyeok":
        webhook_url = "https://hooks.slack.com/services/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"
    payload = {
        "text": "<!channel>",
        "attachments": [{"text": message}]
    }
    headers = {
        "Content-type": "application/json; charset=utf-8"
    }
    response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    # check response is valid
    if response.status_code != 200:
        # warning message
        print("Slack webhook failed with status code: {}".format(response.status_code))
        return False
    return True