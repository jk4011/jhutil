import re
import numpy as np
import lovely_tensors as lt
import json
from argparse import Namespace
from lovely_numpy import lo
from copy import copy, deepcopy
from colorama import Fore, Back, Style

lt.monkey_patch()

disabled = False
diiabled_alphabet = False


def activate_jhprint():
    global disabled, diiabled_alphabet
    disabled = False
    diiabled_alphabet = False


def disable_jhprint(alphabet_only=False):
    global disabled, diiabled_alphabet

    if alphabet_only:
        disabled = False
        diiabled_alphabet = True
    else:
        disabled = True
        diiabled_alphabet = False


def jhprint(key, *datas, yaml=False, list_one_line=True, endline=' ', force=False):
    if force:
        pass
    else:
        if disabled:
            return
        if diiabled_alphabet and re.match(r'[a-z]', str(key)):
            return

    colors = {
        1111: Back.RED + Fore.BLACK,
        2222: Back.YELLOW + Fore.BLACK,
        3333: Back.GREEN + Fore.BLACK,
        4444: Back.CYAN + Fore.BLACK,
        5555: Back.BLUE + Fore.BLACK,
        6666: Fore.RED,
        7777: Fore.YELLOW,
        8888: Fore.GREEN,
        9999: Fore.CYAN,
        0: Back.WHITE + Fore.BLACK,
        "aaaa": Back.RED + Fore.BLACK,
        "bbbb": Back.YELLOW + Fore.BLACK,
        "cccc": Back.GREEN + Fore.BLACK,
        "dddd": Back.CYAN + Fore.BLACK,
        "eeee": Back.BLUE + Fore.BLACK,
        "ffff": Fore.RED,
        "gggg": Fore.YELLOW,
        "hhhh": Fore.GREEN,
        "iiii": Fore.CYAN,
    }
    if isinstance(key, int):
        color = colors[key // 1000 * 1111]
    elif isinstance(key, str):
        color = colors[key]

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
            ret = copy(value)
            if ret.dtype == bool:
                ret = ret.astype(int)
            return lo(ret)
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
    
    if isinstance(key, int):
        if key == 0:
            key = "0000"
        print(color + f"{key} {ret_str}" + Style.RESET_ALL)
    else:
        print(Back.WHITE + key[0:2] + Style.RESET_ALL +
              color + f"{key[2:]} {ret_str}" + Style.RESET_ALL)
