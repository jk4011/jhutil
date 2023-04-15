
from colorama import Back, Style

def jhprint(idx, text=""):
    colors = {
        1111 : Back.RED,
        2222 : Back.YELLOW,
        3333 : Back.GREEN,
        4444 : Back.BLUE,
        5555 : Back.MAGENTA,
        6666 : Back.RED,
        7777 : Back.YELLOW,
        8888 : Back.GREEN,
        9999 : Back.BLUE,
        0 : Back.WHITE,
    }
    color = colors[idx // 1111 * 1111]
    if idx == 0:
        idx = "0000"
    print(color + f"{idx} {text}" + Style.RESET_ALL)


if __name__ == "__main__":
    import jhutil;jhutil.jhprint(0000, )
    import jhutil;jhutil.jhprint(1111, )
    import jhutil;jhutil.jhprint(2222, )
    import jhutil;jhutil.jhprint(3333, )
    import jhutil;jhutil.jhprint(4444, )
    import jhutil;jhutil.jhprint(5555, )
    import jhutil;jhutil.jhprint(6666, )
    import jhutil;jhutil.jhprint(7777, )
    import jhutil;jhutil.jhprint(8888, )
    import jhutil;jhutil.jhprint(9999, )
    

