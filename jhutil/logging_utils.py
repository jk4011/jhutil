import numpy as np
import lovely_tensors as lt
import json
from argparse import Namespace
from lovely_numpy import lo
from copy import deepcopy
from colorama import Fore, Back, Style

lt.monkey_patch()

_disabled = False


def enable_color_log():
    global _disabled
    _disabled = False


def disable_color_log():
    global _disabled
    _disabled = True


def color_log(key, *datas, is_yaml=False, endline=' ', force=False, return_str=False):
    if force:
        pass
    elif _disabled:
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

    ret_str = ""
    for data in datas:
        try:
            if is_yaml:
                data = is_yaml.dump(data, allow_unicode=True,
                                    default_flow_style=False)
            else:  # json
                data = json.dumps(
                    data, indent=4, ensure_ascii=False, default=convert_into_json)
        except:
            pass

        ret_str = ret_str + endline + str(data)

    if isinstance(key, int):
        if key == 0:
            key = "0000"
        output = color + f" {key} {ret_str}" + Style.RESET_ALL
    else:
        output = Back.WHITE + Fore.BLACK + " " + key[0:2] + Style.RESET_ALL + \
            color + f"{key[2:]} {ret_str}" + Style.RESET_ALL

    print(output)
    if return_str:
        return output


def convert_into_json(value):
    value = deepcopy(value)
    if isinstance(value, Namespace):
        return vars(value)
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                value[k] = lo(v)
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            value = value.astype(int)
        return lo(value)
    else:
        return str(value)

