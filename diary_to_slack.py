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
texts = [text for text in texts if len(text) > 0] # remove empty text
num_sends = [text.count("⭐") for text in texts]
texts = [text.replace("⭐", "") for text in texts]

text_weighted = []
for text, num_send in zip(texts, num_sends):
    text_weighted.extend([text] * num_send)


random.shuffle(text_weighted)
print("sending to slack...")
for text in text_weighted:
    rephrased = chatgpt(text)
    send_slack(rephrased, channel="jinhyeok")
    time.sleep(3600)
