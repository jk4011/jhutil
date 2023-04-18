import requests
import json
from .secret import SLACK_WEBHOOK_GPU, SLACK_WEBHOOK_JINHYEOK
from slack_sdk import webhook, WebClient

def send_slack(message, channel="gpu"):
    if channel == "gpu":
        webhook_url = SLACK_WEBHOOK_GPU
    elif channel == "jinhyeok":
        webhook_url = SLACK_WEBHOOK_JINHYEOK
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

def slack_wrapper(func, *args, **kwargs):
    try:
        import jhutil;jhutil.jhprint(0000, "started slack_wrapper")
        
        # run function
        func(*args, **kwargs)
        
        message = "Finished!"
        jhutil.send_slack(message)
        import jhutil;jhutil.jhprint(3333, message)

    except Exception as e:
        error_message = f"Error: ```{str(e)}```"
        jhutil.send_slack(error_message)
        import jhutil;jhutil.jhprint(1111, error_message)