import torch
import numpy as np
from jhutil.logging_utils import color_log, disable_color_log, activate_color_log
from jhutil.logging_utils import *

def jhprint_old(key, *datas, yaml=False, list_one_line=True, endline=' ', force=False):
    if force:
        pass

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
        value = deepcopy(value)
        if isinstance(value, Namespace):
            return vars(value)
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    value[k] = lo(v)
            return value
        if isinstance(value, (list, tuple)):
            if list_one_line:
                return str(value)
            else:
                return value
        if isinstance(value, np.ndarray):
            if value.dtype == bool:
                value = value.astype(int)
            return lo(value)
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
        output = color + f"{key} {ret_str}" + Style.RESET_ALL
    else:
        output = Back.WHITE + key[0:2] + Style.RESET_ALL + \
              color + f"{key[2:]} {ret_str}" + Style.RESET_ALL
    
    print(output)
    return output

def test_color_log():

    tensor = torch.randn(100, 100)
    tensor_list = [torch.randn(100, 100), torch.randn(100, 100), torch.randn(100, 100)]
    array = np.random.randn(100, 100)
    array_list = [np.random.randn(100, 100), np.random.randn(100, 100), np.random.randn(100, 100)]
    
    datas = [tensor, tensor_list, array, array_list]
    
    for data in datas:
        ouptut1 = jhprint_old(1111, tensor)
        ouptut2 = jhprint_old(2222, tensor, list_one_line=False)
        ouptut3 = jhprint_old(3333, tensor, endline='\n ')
        
        assert(ouptut1 == color_log(1111, tensor))
        assert(ouptut2 == color_log(2222, tensor, list_in_one_line=False))
        assert(ouptut3 == color_log(3333, tensor, endline='\n '))
