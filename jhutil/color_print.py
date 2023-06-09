import re
import numpy as np
import lovely_tensors as lt
import json
from argparse import Namespace
from lovely_numpy import lo
from copy import copy, deepcopy
from colorama import Fore, Back, Style

lt.monkey_patch()


def jhprint(idx, *datas, yaml=False, list_one_line=True, endline=' '):
    colors = {
        1111: Back.RED,
        2222: Back.YELLOW,
        3333: Back.GREEN,
        4444: Back.BLUE,
        5555: Back.MAGENTA,
        6666: Fore.RED,
        7777: Fore.YELLOW,
        8888: Fore.GREEN,
        9999: Fore.BLUE,
        0: Back.WHITE,
    }

    color = colors[idx // 1000 * 1111]

    def json_default(value):
        if isinstance(value, Namespace):
            return vars(value)
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    value[k] = lo(copy(v))
            return value
        if isinstance(value, (list, tuple)):
            if list_one_line:
                return str(value)
            else:
                return value
        if isinstance(value, np.ndarray):
            return lo(copy(value))
        else:
            return str(value)

    ret_str = ""
    for data in datas:
        # make pretty
        try:
            if yaml:
                data = yaml.dump(data, allow_unicode=True,
                                 default_flow_style=False)
            else:  # json
                data = json_default(data)
                data = json.dumps(
                    data, indent=4, ensure_ascii=False, default=json_default)
        except:
            pass

        ret_str = ret_str + endline + str(data)

    # for 0000
    if idx == 0:
        idx = "0000"

    print(color + f"{idx} {ret_str}" + Style.RESET_ALL)
