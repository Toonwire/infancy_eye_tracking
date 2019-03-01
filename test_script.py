#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:33:14 2019

@author: sebastiannyholm
"""

from PIL import Image, ImageTk



image = Image.open("stimuli/original-pony-dancing.gif")

try:
    while True:
        print(image.tell())
        image.seek(image.tell()+1)
        
except EOFError: pass



#image = image.resize((int(self.width), int(self.height)), Image.ANTIALIAS) #The (250, 250) is (height, width)
