
from colorama import Fore, Back, Style
import json

# make pretty tensor
import lovely_tensors as lt
lt.monkey_patch()

from argparse import Namespace
import re
import numpy as np
from lovely_numpy import lo


def jhprint(idx, *datas, yaml=False, list_one_line=True, endline=' '):
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
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)):
            if list_one_line:
                return str(value)
            else:
                return value
        if isinstance(value, np.ndarray):
            return lo(value)
        else:
            return str(value)

    ret_str = ""
    for data in datas:
        # make pretty
        try:
            if yaml:
                data = yaml.dump(data, allow_unicode=True, default_flow_style=False)
            else: # json
                data = json_default(data)
                data = json.dumps(data, indent=4, ensure_ascii=False)
        except:
            pass
        ret_str = ret_str + endline + str(data)
    
    # for 0000
    if idx == 0:
        idx = "0000"

    print(color + f"{idx} {ret_str}" + Style.RESET_ALL)



