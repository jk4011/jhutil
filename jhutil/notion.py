import requests
import os
from notion_client import Client
from pprint import pprint

from .secret import NOTION_TOKEN, NOTION_DATABASE_ID_DIARY, NOTION_DATABASE_ID_SCRATCH
from .slack import send_slack
from .chatgpt import chatgpt

import time


notion = Client(auth=NOTION_TOKEN)

from notion_client.helpers import is_full_page


def parse_simple_page(page_id):
    """Parse simple notion page. 
    The blocks in page must not have any child block.
    
    For example, suppose a notion page like this:
    ## 주제
    내용
    
    Then the output is like:
    [
        ['heading_2', '주제'], 
        ['paragraph', '내용'],
    ]
    """
    blocks = notion.blocks.children.list(page_id)
    page_data = []
    for block in blocks["results"]:
        type = block["type"]

        # skip empty block
        if len(block[type]["rich_text"]) == 0:
            continue
        
        text = block[type]["rich_text"][0]["plain_text"]

        result = [type, text]
        page_data.append(result)
        
    return page_data

def get_very_recent_page(database_id):
    full_or_partial_pages = notion.databases.query(
        database_id=database_id,
    )
    pages = full_or_partial_pages["results"]
    very_recent_page_id = pages[0]["id"]
    page_data = parse_simple_page(very_recent_page_id)

    return page_data


def diary_to_slack(use_chatgpt=True):
    page_data = get_very_recent_page(database_id=NOTION_DATABASE_ID_DIARY)

    texts = ""
    for type, text in page_data:
        
        if type.startswith("heading"):
            texts = texts + "[SPLIT_TOKEN]" + "## " + text + "\n\n"
        else:
            texts = texts + text + "\n"
        
    texts = texts.split("[SPLIT_TOKEN]")
    texts = [text for text in texts if len(text) > 0] # remove empty text

    if use_chatgpt:
        texts = [chatgpt(text) for text in texts]

    for text in texts:
        send_slack(text, channel="jinhyeok")
        time.sleep(3600)
