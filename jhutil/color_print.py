
from colorama import Fore, Back, Style
import json

# make pretty tensor
import lovely_tensors as lt
lt.monkey_patch()

from argparse import Namespace
import re


def jhprint(idx, data="", yaml=False, list_one_line=True):
    colors = {
        1111 : Back.RED,
        2222 : Back.YELLOW,
        3333 : Back.GREEN,
        4444 : Back.BLUE,
        5555 : Back.MAGENTA,
        6666 : Fore.RED,
        7777 : Fore.YELLOW,
        8888 : Fore.GREEN,
        9999 : Fore.BLUE,
        0 : Back.WHITE,
    }
    color = colors[idx // 1000 * 1111]

    
    def json_default(value):
        if isinstance(value, Namespace):
            return vars(value)
        if isinstance(value, list):
            return str(value)
        else:
            return str(value)

    # make pretty dictionaryls
    if isinstance(data, dict):
        
        try:
            if yaml:
                data = yaml.dump(data, allow_unicode=True, default_flow_style=False)
            else: # json
                data = json.dumps(data, indent=4, ensure_ascii=False, default=json_default)
            if list_one_line:
                # data = re.sub(r'",\s+', '", ', data)
                data = re.sub(r'(\d),\s+', r'\1, ', data)
        except:
            print("hello")
            pass
    
    # for 0000
    if idx == 0:
        idx = "0000"

    print(color + f"{idx} {data}" + Style.RESET_ALL)


