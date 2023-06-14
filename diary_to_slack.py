import time
import random
from jhutil import get_very_recent_page, NOTION_DATABASE_ID_DIARY, chatgpt, send_slack

print("retreive page data from notion...")
page_data = get_very_recent_page(database_id=NOTION_DATABASE_ID_DIARY)

texts = ""
for type, text in page_data:
    if type.startswith("heading"):
        texts = texts + "[SPLIT_TOKEN]" + "## " + text + "\n\n"
    else:
        texts = texts + text + "\n"

texts = texts.split("[SPLIT_TOKEN]")
texts = [text for text in texts if len(text) > 0]  # remove empty text

gpt_inputs = [text for text in texts if "⭐" in text]
pure_texts = [text for text in texts if "📩" in text]

texts = [text.replace("⭐", "") for text in texts]

pure_weighted = []
for text in pure_texts:
    num_send = text.count("📩")
    pure_weighted.extend([text.replace("📩", "")] * num_send)


gpt_weighted = []
for text in gpt_inputs:
    num_send = text.count("⭐")
    gpt_weighted.extend([text.replace("⭐", "")] * num_send)

print("sending pure messages...")
for text in pure_weighted:
    send_slack(text, channel="jinhyeok")
    time.sleep(3600)

print("sending gpt messages...")
for text in gpt_weighted:
    rephrased = chatgpt(text)
    send_slack(rephrased, channel="jinhyeok")
    time.sleep(3600)
