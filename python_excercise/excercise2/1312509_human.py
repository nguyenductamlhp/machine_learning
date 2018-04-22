# -*- encoding: utf-8 -*-
import sys
from datetime import datetime
import re
import json


class Human:

    def __init__(self, name=None, birthyear=None):
        self.name = name
        self.birthyear = birthyear

    def read_from_json(self, file_name):
        data = json.load(open(file_name))
        self.name = data['Name']
        self.birthyear = data['BirthYear']

    def get_age(self):
        td = datetime.now().date()
        cur_year = td.year
        return cur_year - self.birthyear

    def show_info(self):
        print "Name: ", self.name
        print "Birth Year: ", self.birthyear
        print "Age: ", self.get_age()
        
    
def main(argv):
    try:
        with open(argv[1]) as file:
            human = Human(None, None)
            human.read_from_json(argv[1])
            human.show_info()
    except IOError:
        print "Error when read file!"

if __name__ == "__main__":
    print "Read and display human infomation ..."
    main(sys.argv[:])