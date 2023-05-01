from jhutil import get_very_recent_page, NOTION_DATABASE_ID_DIARY, chatgpt, send_slack
import time


import jhutil;jhutil.jhprint(1111, "retreive page data...")
page_data = get_very_recent_page(database_id=NOTION_DATABASE_ID_DIARY)

texts = ""
for type, text in page_data:
    if type.startswith("heading"):
        texts = texts + "[SPLIT_TOKEN]" + "## " + text + "\n\n"
    else:
        texts = texts + text + "\n"
    
texts = texts.split("[SPLIT_TOKEN]")
texts = [text for text in texts if len(text) > 0] # remove empty text


import jhutil;jhutil.jhprint(2222, "sending slack...")
for text in texts:
    rephrased = chatgpt(text)
    send_slack(rephrased, channel="jinhyeok")
    time.sleep(3600)
