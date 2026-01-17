import numpy as np
import lovely_tensors as lt
import json
from argparse import Namespace
from lovely_numpy import lo
from copy import deepcopy
from colorama import Fore, Back, Style
import traceback
import logging
import os
from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from contextlib import nullcontext


lt.monkey_patch()
location_history = {}


def silent_fn(stdout=True, stderr=False):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with open(os.devnull, "w") as devnull:
                with (redirect_stdout(devnull) if stdout else nullcontext()), \
                     (redirect_stderr(devnull) if stderr else nullcontext()):
                    return fn(*args, **kwargs)
        return wrapper
    return deco



def file_log(file_name, *datas):    
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # if this logger is not initialized, initialize it
    if not logger.handlers:
        fh = logging.FileHandler(file_name, encoding="utf-8")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    print(datas)
    logger.info(" ".join(str(data) for data in datas))



def color_log(key, *datas, is_yaml=False, endline=' ', return_str=False, repeat=True, update=False):
    
    if _traceback_enabled:
        stack = traceback.extract_stack()
        filename, lineno, funcname, code = stack[-2]
        print(f"Called from file: {filename}, line {lineno}")
        
    if not repeat:
        stack = traceback.extract_stack()
        filename, lineno, funcname, code = stack[-2]
        location = f"{filename}:{lineno}"
        
        if location in location_history:
            return
        else:
            location_history[location] = True

    
    colors = {
        1111: Back.RED + Fore.BLACK,
        2222: Back.YELLOW + Fore.BLACK,
        3333: Back.GREEN + Fore.BLACK,
        4444: Back.CYAN + Fore.BLACK,
        5555: Back.BLUE + Fore.BLACK,
        6666: Back.RED + Fore.BLACK,
        7777: Back.YELLOW + Fore.BLACK,
        8888: Back.GREEN + Fore.BLACK,
        9999: Back.CYAN + Fore.BLACK,
        0: Back.WHITE + Fore.BLACK,
        "aaaa": Fore.RED,
        "bbbb": Fore.YELLOW,
        "cccc": Fore.GREEN,
        "dddd": Fore.CYAN,
        "eeee": Fore.BLUE,
        "ffff": Fore.RED,
        "gggg": Fore.YELLOW,
        "hhhh": Fore.GREEN,
        "iiii": Fore.CYAN,
    }
    key_type = type(key)
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

    if key in [6666, 7777, 8888, 9999, "ffff", "gggg", "hhhh", "iiii"]:
        key = str(key)
        output = Back.WHITE + Fore.BLACK + " " + key[0:1] + Style.RESET_ALL + \
            color + f"{key[1:]} {ret_str}" + Style.RESET_ALL
    else:
        if key == 0:
            key = "0000"
        output = color + f" {key} {ret_str}" + Style.RESET_ALL

    if key_type == str:
        output = "  " + output
    if update:
        print('\r' + output, end='')
    else:
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


def clear_location_history():
    global location_history
    location_history = {}


_traceback_enabled = False


def enable_traceback():
    global _traceback_enabled
    _traceback_enabled = True


def disable_traceback():
    global _traceback_enabled
    _traceback_enabled = False