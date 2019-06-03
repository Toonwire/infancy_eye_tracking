# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:17:49 2019

@author: Toonw
"""
import numpy as np

def vlen(a):
    return (a[0]**2 + a[1]**2)**0.5

def add(v1,v2):
    return (v1[0]+v2[0], v1[1]+v2[1])

def sub(v1,v2):
    return (v1[0]-v2[0], v1[1]-v2[1])



def unit_vector(v):
    vu =  v / np.linalg.norm(v)
    return (vu[0], vu[1])

def angle_between(v1, v2):
    angle = np.arccos(np.dot(v1,v2)/(vlen(v1)*vlen(v2)))
    return angle
        
# Similarity measure of article
## https://pdfs.semanticscholar.org/60b5/aca20ba34d424f4236359bd5e6aa30487682.pdf
def sim_measure(A, B): # similarity between two shapes A and B
#    print(A)
#    print(B)
    return 1 - (sum([(vlen(unit_vector(a))+vlen(unit_vector(b)))*angle_between(a,b) for a,b in zip(A,B)]))/(np.pi*(len(A)+len(B)))