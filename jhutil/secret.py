# enviroment variable for secret key
import os

OPENAI_APIKEY = os.environ.get('OPENAI_APIKEY')
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")

SLACK_WEBHOOK_PROCESS = os.environ.get("PROCESS_WEBHOOK")
SLACK_WEBHOOK_JINHYEOK = os.environ.get("JINHYEOK_WEBHOOK")
channel_to_webhook_url = {
    "gpu": SLACK_WEBHOOK_PROCESS,
    "jinhyeok": SLACK_WEBHOOK_JINHYEOK
}