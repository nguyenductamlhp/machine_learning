# -*- encoding: utf-8 -*-
import sys
from datetime import datetime
import re
import json

# PNG, JPG, GIF, BMP

def main(argv):
    try:
        with open(argv[1]) as f:
            byte = f.read(1)
            while byte != "":
                # Do stuff with byte.
                byte = f.read(1)
                print ">>> byte", byte
    except IOError:
        print "Error when read file!"

if __name__ == "__main__":
    print "Read and display human infomation ..."
    main(sys.argv[:])