import time
from contextlib import contextmanager
from jhutil import color_log

class print_time:
    def __init__(self, func_name=""):
        self.func_name = func_name
        self.start_time = None

    # 1️⃣ with문용 메서드
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()        
        color_log("aaaa", f"{self.func_name} took {end_time - self.start_time:.2f} seconds")

    # 2️⃣ decorator용 메서드
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.__class__(func.__name__ or self.func_name):
                return func(*args, **kwargs)
        return wrapper
