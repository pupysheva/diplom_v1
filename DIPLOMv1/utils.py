import time

from functools import wraps
from math import trunc


def timer(text=''):
    """Декоратор, печатает время выполнения декорированной функции.
    Args:
        text строка: текст для печати до отображения времени.
    """
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            hours = trunc((end - start) / 3600)
            minutes = trunc((end - start) / 60)
            seconds = round((end - start) % 60)

            if hours > 1:
                print(text + '{} hours {} min and {} sec'.format(hours, minutes,
                                                                 seconds))
            elif hours == 1:
                print(text + '{} hour {} min and {} sec'.format(hours, minutes,
                                                                seconds))
            elif minutes >= 1:
                print(text + '{} min and {} sec'.format(minutes, seconds))
            else:
                print(text + '{} sec'.format(seconds))

            return result

        return wrapper

    return decorator