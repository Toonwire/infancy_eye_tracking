#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:13:44 2019

@author: sebastiannyholm
"""

import numpy as np
import math

import psychopy.visual
import psychopy.event
import psychopy.core
import psychopy.monitors


try:
    import Image
    import ImageDraw
except:
    from PIL import Image
    from PIL import ImageDraw


start = (-0.7,-0.5)
end = (0.0,0.0)

r = abs(((end[0] - start[0])**2+(end[1] - start[1])**2)**0.5)

theta_x = math.acos(start[0]/r)
theta_y = math.asin(start[1]/r)

theta = theta_x
if theta_y < 0:
    theta = -theta_x

img_intermediate_positions = [start]

while r >= 0:
    r -= 0.015
    theta = theta + 0.05 * math.pi
    
    pos = (r*math.cos(theta), r*math.sin(theta))
    
    img_intermediate_positions.append(pos)



#--------------------------------------------


mon = psychopy.monitors.Monitor('MyScreen')

# get distance to dcreen from config
mon.setSizePix((800, 600))

win = psychopy.visual.Window(size=(800, 600), fullscr=True, units='norm', monitor=mon, rgb=(1,1,1))




psychopy.event.Mouse(visible=False, win=win)
        
background = Image.new('RGBA',tuple(win.size))
ImageDraw.Draw(background)


img = Image.open("stimuli/star_yellow.png")
stimuli = psychopy.visual.ImageStim(win, image=img, autoLog=False)
stimuli.size = (0.2,0.2)

for i, img_pos in enumerate(img_intermediate_positions):
    
    stimuli.setPos(img_pos)
    stimuli.ori = i*20
    stimuli.draw()
    win.flip()
    
    psychopy.core.wait(0.1)
        



win.winHandle.set_fullscreen(False) # disable fullscreen
win.close()