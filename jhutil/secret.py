# enviroment variable for secret key
import os

SLACK_WEBHOOK_PROCESS = os.environ.get("PROCESS_WEBHOOK")
SLACK_WEBHOOK_JINHYEOK = os.environ.get("JINHYEOK_WEBHOOK")
OPENAI_APIKEY = os.environ.get('OPENAI_APIKEY')
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID_DIARY = os.environ.get("NOTION_DATABASE_ID_DIARY")
NOTION_DATABASE_ID_SCRATCH = os.environ.get("NOTION_DATABASE_ID_SCRATCH")
