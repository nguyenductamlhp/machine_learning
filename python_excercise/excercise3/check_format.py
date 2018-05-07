# -*- encoding: utf-8 -*-
import sys
from datetime import datetime
import re
import json

# PNG, JPG, GIF, BMP
patt = {
    'BM': 'BMP',
    'PNG': 'PNG',
    'GIF': 'GIF',
    'JFIF': 'JPG'
}

def main(argv):
    try:
        with open(argv[1]) as f:
            data = f.read(100)
            for k in patt.keys():
                if k in data:
                    print patt[k]
                    return
    except IOError:
        print "Error when read file!"

if __name__ == "__main__":
    main(sys.argv[:])
