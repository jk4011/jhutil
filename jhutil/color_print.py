
from colorama import Fore, Back, Style
import json

# make pretty tensor
import lovely_tensors as lt
lt.monkey_patch()


def jhprint(idx, text=""):
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

    # make pretty dictionaryls
    if isinstance(text, dict):
        try:
            text = json.dumps(text, indent=4, ensure_ascii=False)
        except:
            pass
    
    # for 0000
    if idx == 0:
        idx = "0000"

    print(color + f"{idx} {text}" + Style.RESET_ALL)


