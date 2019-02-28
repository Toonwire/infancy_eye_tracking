# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:06:20 2019

@author: s144451
"""

import math
from scipy import optimize
import numpy as np


class DataCorrection:
    
    transformation_matrix_left_eye = np.identity(2)
    transformation_matrix_right_eye = np.identity(2)
    
    def __init__(self, targets, px_width, px_height):
        self.targets = targets
        self.px_width = px_width
        self.px_height = px_height
    
    def avg_dist_to_closest_fixation(self, transformation):
        transformation = np.reshape(transformation, (-1,2))
        coords = np.matmul(transformation, self.calibration_fixations)
        distClosest = np.zeros(len(coords[0,:]))
        
        for fixNum in range(len(coords[0,:])):
            distClosest[fixNum] = math.sqrt((coords[0,fixNum]-self.targets[0,fixNum])**2 + (coords[1,fixNum]-self.targets[1,fixNum])**2)
            
        avgDistance = np.mean(distClosest)
        return avgDistance
    
    def calibrate_left_eye(self, fixations):
        self.calibration_fixations = fixations
        print("Calibrating left eye\n----------------")
        self.transformation_matrix_left_eye = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        self.transformation_matrix_left_eye = np.reshape(self.transformation_matrix_left_eye, (-1,2))
        
    def calibrate_right_eye(self, fixations):
        self.calibration_fixations = fixations
        print("Calibrating right eye\n----------------")
        self.transformation_matrix_right_eye = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        self.transformation_matrix_right_eye = np.reshape(self.transformation_matrix_right_eye, (-1,2))
        
    
    def adjust_left_eye(self, fixations):
        if np.allclose(self.transformation_matrix_left_eye, np.identity(2)):
            raise Exception("No calibration for left eye exists")
        return np.matmul(self.transformation_matrix_left_eye, fixations)
        
    def adjust_right_eye(self, fixations):
        if np.allclose(self.transformation_matrix_right_eye, np.identity(2)):
            raise Exception("No calibration for right eye exists")
        return np.matmul(self.transformation_matrix_right_eye, fixations)
        