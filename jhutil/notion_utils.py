from notion_client import Client
from .secret import NOTION_TOKEN
import time


notion = Client(auth=NOTION_TOKEN)


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
        if "rich_text" not in block[type]:
            continue
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
