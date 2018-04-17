# -*- encoding: utf-8 -*-
import sys
from datetime import datetime
from datetime import timedelta
import pytz
from pytz import timezone
import re

def main(argv):
    '''
    Return time of postion inputted
    >>> python world_time.py Ho Chi Minh
    09:15:24 17-04-2018
    >>> python world_time.py San franCisco
    19:26:22 16/04/2018
    >>> python world_time.py Hanoi
    None
    '''
    tz_data = {
        'ho chi minh': 7,
        'san francisco': -7,
        'paris': 2,
        'frankfurt': 1,
        'greece': 2,
        'taipei': 8,
        'tokyo': 9,
        'brisbane': 10,
        'singapore': 8,
        'beijing': 8,
        'seoul': 9,
    }
    fmt = '%H:%M:%S %d/%m/%Y'
    position = ' '.join(argv).lower()
    if position in tz_data.keys():
        dt = None
        dt = datetime.now() + timedelta(hours=tz_data[position])
        return dt.strftime(fmt)
    return None

if __name__ == "__main__":
    print main(sys.argv[1:])
