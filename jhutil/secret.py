# enviroment variable for secret key
import os

OPENAI_APIKEY = os.environ.get('OPENAI_APIKEY')
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")

channel_to_webhook_url = {
    "gpu": os.environ.get("PROCESS_WEBHOOK"),
    "jinhyeok": os.environ.get("JINHYEOK_WEBHOOK")
}