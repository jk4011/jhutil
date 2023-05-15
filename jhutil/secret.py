# enviroment variable for secret key
import os

SLACK_WEBHOOK_GPU = "https://hooks.slack.com/services/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"
SLACK_WEBHOOK_JINHYEOK = os.environ.get("SLACK_WEBHOOK_JINHYEOK")
OPENAI_APIKEY = os.environ.get('OPENAI_APIKEY')
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID_DIARY = os.environ.get("NOTION_DATABASE_ID_DIARY")
NOTION_DATABASE_ID_SCRATCH = os.environ.get("NOTION_DATABASE_ID_SCRATCH")
