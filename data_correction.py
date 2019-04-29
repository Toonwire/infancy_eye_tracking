# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:06:20 2019

@author: s144451
"""

from scipy import optimize
import numpy as np
import sys


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
        
        distClosest = []
        
        for f, t in zip(coords.T, self.calibration_targets.T):
            distClosest.append(self.euclidean_distance(f,t))
                        
        return np.mean(distClosest)
    
    def calibrate_left_eye(self, fixations, initial_guess=np.identity(2)):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating left eye\n----------------")
        self.transformation_matrix_left_eye = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=initial_guess)
        self.transformation_matrix_left_eye = np.reshape(self.transformation_matrix_left_eye, (-1,2))
        print("")
        
    def calibrate_right_eye(self, fixations, initial_guess=np.identity(2)):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating right eye\n----------------")
        self.transformation_matrix_right_eye = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=initial_guess)
        self.transformation_matrix_right_eye = np.reshape(self.transformation_matrix_right_eye, (-1,2))
        print("")
                
    def adjust_left_eye(self, fixations):
        if np.allclose(self.transformation_matrix_left_eye, np.identity(2)):
            raise Exception("No calibration for left eye exists")
        return np.matmul(self.transformation_matrix_left_eye, fixations)
        
    def adjust_right_eye(self, fixations):
        if np.allclose(self.transformation_matrix_right_eye, np.identity(2)):
            raise Exception("No calibration for right eye exists")
        return np.matmul(self.transformation_matrix_right_eye, fixations)
    
    
    transformation_matrix_left_eye_poly = np.ones((2,3))
    transformation_matrix_right_eye_poly = np.ones((2,3))
    poly_init_matrix = np.array([[0,0,0],[0,0,0]])
    
    def apply_polynomial(self, fixations, transformation):

        cor_x = []
        cor_y = []
        
        for x, y in fixations.T:
            
            x_cor = x
            y_cor = y
            
            for i, (a, b) in enumerate(transformation.T):
                
                x_cor += a * y**i
                y_cor += b * x**i
            
            cor_x.append(x_cor)
            cor_y.append(y_cor)
        
        return np.array([cor_x, cor_y])
    
    def avg_dist_to_closest_fixation_poly(self, transformation):
            
        transformation = np.reshape(transformation, (2,-1))
        fixations_cor = self.apply_polynomial(self.calibration_fixations, transformation)
        
        distClosest = []
        
        for f, t in zip(fixations_cor.T, self.calibration_targets.T):            
            distClosest.append(self.euclidean_distance(f,t))
            
        return np.mean(distClosest)
    
    def calibrate_left_eye_poly(self, fixations):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating left eye\n----------------")
        self.transformation_matrix_left_eye_poly = optimize.fmin(func=self.avg_dist_to_closest_fixation_poly, x0=self.poly_init_matrix)
        self.transformation_matrix_left_eye_poly = np.reshape(self.transformation_matrix_left_eye_poly, (2,-1))
        
    def calibrate_right_eye_poly(self, fixations):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating right eye\n----------------")
        self.transformation_matrix_right_eye_poly = optimize.fmin(func=self.avg_dist_to_closest_fixation_poly, x0=self.poly_init_matrix)
        self.transformation_matrix_right_eye_poly = np.reshape(self.transformation_matrix_right_eye_poly, (2,-1))
    
    def adjust_left_eye_poly(self, fixations):
        if np.allclose(self.transformation_matrix_left_eye_poly, self.poly_init_matrix):
            raise Exception("No calibration for left eye exists")
        
        return self.apply_polynomial(fixations, self.transformation_matrix_left_eye_poly)
        
    def adjust_right_eye_poly(self, fixations):
        if np.allclose(self.transformation_matrix_right_eye_poly, self.poly_init_matrix):
            raise Exception("No calibration for left eye exists")
        
        return self.apply_polynomial(fixations, self.transformation_matrix_right_eye_poly)

        
    def get_left_poly_coeffs(self):
        return (self.transformation_matrix_left_eye_poly[0,0], self.transformation_matrix_left_eye_poly[0,1], self.transformation_matrix_left_eye_poly[0,2],
                self.transformation_matrix_left_eye_poly[1,0], self.transformation_matrix_left_eye_poly[1,1], self.transformation_matrix_left_eye_poly[1,2]) 
              
    def get_right_poly_coeffs(self):
        return (self.transformation_matrix_right_eye_poly[0,0], self.transformation_matrix_right_eye_poly[0,1], self.transformation_matrix_right_eye_poly[0,2],
                self.transformation_matrix_right_eye_poly[1,0], self.transformation_matrix_right_eye_poly[1,1], self.transformation_matrix_right_eye_poly[1,2]) 
        
    # Scan all points 
    # Compute distance and check eps
    # Add to result
    def range_query_linear(self, points, p, eps):
        
        neighbors = set()
        
        misses = 0
        for i in range(p[2]-1, -1, -1):
            
            q = points[i]
            
            if p != q and self.euclidean_distance(p,q) <= eps:
                neighbors.add(q)
                misses = 0
            else:
                misses += 1

            if misses == 3:
                break
            
        return neighbors
        
    def adjust_by_cluster_center(self, fixations):
        
        cor_x = []
        cor_y = []

        points = [(p[0], p[1], i) for i, p in enumerate(fixations.T)] # convert to tuples 
        eps = 0.05
        
        clusters = []
        
        for i in range(len(points)):
            p = points[i]
            
            cluster = self.range_query_linear(points, p, eps)  # Find more neighbors
            
            new_cluster = set()
            new_cluster.add(p)
            new_cluster.update(cluster)
            for q in cluster:                
                new_cluster.update(clusters[q[2]])

            clusters.append(new_cluster)
            
            mean_x = sum([c[0] for c in new_cluster]) / len(new_cluster)
            mean_y = sum([c[1] for c in new_cluster]) / len(new_cluster)
            
            cor_x.append(mean_x)
            cor_y.append(mean_y)
            
        return np.array([cor_x,cor_y])
        
    
    def coef_func(self, transformation):
        transformation = np.reshape(transformation, (2,-1))
        a0 = transformation[0,0]
        a1 = transformation[0,1]
        a2 = transformation[0,2]
        a3 = transformation[0,3]
        a4 = transformation[0,4]
        a5 = transformation[0,5]
        
        b0 = transformation[1,0]
        b1 = transformation[1,1]
        b2 = transformation[1,2]
        b3 = transformation[1,3]
        b4 = transformation[1,4]
        b5 = transformation[1,5]
        
        
        distClosest = np.zeros(len(self.calibration_fixations[0,:]))        
        
        # x' = a0 + a1x + a2x^2 + a3y + a4y^2 + a5xy
        # y' = b0 + b1x + b2x^2 + b3y + b4y^2 + b5xy
        
        for i in range(len(self.calibration_fixations[0,:])):
            x = self.calibration_fixations[0,i]
            y = self.calibration_fixations[1,i]
            
            corrected_x = a0 + a1*x + a2*x**2 + a3*y + a4*y**2 + a5*x*y
            corrected_y = b0 + b1*x + b2*x**2 + b3*y + b4*y**2 + b5*x*y           
        
            distClosest[i] = ((corrected_x-self.calibration_targets[0,i])**2 + (corrected_y-self.calibration_targets[1,i])**2) ** 0.5
        
        avgDistance = np.mean(distClosest)
        return avgDistance
        
        
        
        
    
    def calibrate_left_eye_coef(self, fixations, initial_guess=np.ones((2,6))):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating left eye\n----------------")
        self.transformation_matrix_left_eye = optimize.fmin(func=self.coef_func, x0=initial_guess, xtol=0.3)
        self.transformation_matrix_left_eye = np.reshape(self.transformation_matrix_left_eye, (2,-1))
        
    def calibrate_right_eye_coef(self, fixations, initial_guess=np.ones((2,6))):
        self.calibration_fixations = fixations
        self.calibration_targets = self.targets
        print("Calibrating right eye\n----------------")
        self.transformation_matrix_right_eye = optimize.fmin(func=self.coef_func, x0=initial_guess, xtol=0.3)
        self.transformation_matrix_right_eye = np.reshape(self.transformation_matrix_right_eye, (2,-1))
    
        
        
    def adjust_left_eye_coef(self, fixations):
        if np.allclose(self.transformation_matrix_left_eye, np.ones((2,6))):
            raise Exception("No calibration for left eye exists")            
        return self.apply_coefs(self.transformation_matrix_left_eye, fixations)
        
    def adjust_right_eye_coef(self, fixations):
        if np.allclose(self.transformation_matrix_right_eye, np.ones((2,6))):
            raise Exception("No calibration for right eye exists")
        return self.apply_coefs(self.transformation_matrix_right_eye, fixations)
    
    
    def apply_coefs(self, transformation, fixations):
        a0 = transformation[0,0]
        a1 = transformation[0,1]
        a2 = transformation[0,2]
        a3 = transformation[0,3]
        a4 = transformation[0,4]
        a5 = transformation[0,5]
        
        b0 = transformation[1,0]
        b1 = transformation[1,1]
        b2 = transformation[1,2]
        b3 = transformation[1,3]
        b4 = transformation[1,4]
        b5 = transformation[1,5]
        
        corrected_x = []
        corrected_y = []
        for i in range(len(fixations[0,:])):
            x = fixations[0,i]
            y = fixations[1,i]
            corrected_x.append(a0 + a1*x + a2*x**2 + a3*y + a4*y**2 + a5*x*y)
            corrected_y.append(b0 + b1*x + b2*x**2 + b3*y + b4*y**2 + b5*x*y)
        
        return np.array([corrected_x, corrected_y])
        
        

    # Fun with points
    transformation_matrices_left_eye = {}
    transformation_matrices_left_eye["upper_right"] = np.identity(2)
    transformation_matrices_left_eye["upper_left"] = np.identity(2)
    transformation_matrices_left_eye["bottom_right"] = np.identity(2)
    transformation_matrices_left_eye["bottom_left"] = np.identity(2)
    transformation_matrices_left_eye["center"] = np.identity(2)
    
    transformation_matrices_right_eye = {}
    transformation_matrices_right_eye["upper_right"] = np.identity(2)
    transformation_matrices_right_eye["upper_left"] = np.identity(2)
    transformation_matrices_right_eye["bottom_right"] = np.identity(2)
    transformation_matrices_right_eye["bottom_left"] = np.identity(2)
    transformation_matrices_right_eye["center"] = np.identity(2)    
    
    target_center = None
    target_upper_left = None
    target_upper_right = None
    target_bottom_left = None
    target_bottom_right = None
    
    def calibrate_left_eye_seb(self, fixations):
        
        fixation_upper_right, fixation_upper_left, fixation_bottom_right, fixation_bottom_left, fixation_center, target_points_upper_right, target_points_upper_left, target_points_bottom_right, target_points_bottom_left, target_points_center = self.seperate_fixations(fixations)
            
        print("Calibrating left eye\n----------------")
        self.calibration_fixations = fixation_upper_right
        self.calibration_targets = target_points_upper_right
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_left_eye["upper_right"] = trans_matrix
        
        self.calibration_fixations = fixation_upper_left
        self.calibration_targets = target_points_upper_left
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_left_eye["upper_left"] = trans_matrix
        
        self.calibration_fixations = fixation_bottom_right
        self.calibration_targets = target_points_bottom_right
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_left_eye["bottom_right"] = trans_matrix
        
        self.calibration_fixations = fixation_bottom_left
        self.calibration_targets = target_points_bottom_left
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_left_eye["bottom_left"] = trans_matrix
        
        self.calibration_fixations = fixation_center
        self.calibration_targets = target_points_center
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_left_eye["center"] = trans_matrix
        

    def calibrate_right_eye_seb(self, fixations):
        fixation_upper_right, fixation_upper_left, fixation_bottom_right, fixation_bottom_left, fixation_center, target_points_upper_right, target_points_upper_left, target_points_bottom_right, target_points_bottom_left, target_points_center = self.seperate_fixations(fixations)
            
        print("Calibrating left eye\n----------------")        
        self.calibration_fixations = fixation_upper_right
        self.calibration_targets = target_points_upper_right
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_right_eye["upper_right"] = trans_matrix
        
        self.calibration_fixations = fixation_upper_left
        self.calibration_targets = target_points_upper_left
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_right_eye["upper_left"] = trans_matrix
        
        self.calibration_fixations = fixation_bottom_right
        self.calibration_targets = target_points_bottom_right
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_right_eye["bottom_right"] = trans_matrix
        
        self.calibration_fixations = fixation_bottom_left
        self.calibration_targets = target_points_bottom_left
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_right_eye["bottom_left"] = trans_matrix
        
        self.calibration_fixations = fixation_center
        self.calibration_targets = target_points_center
        trans_matrix = optimize.fmin(func=self.avg_dist_to_closest_fixation, x0=np.identity(2))
        trans_matrix = np.reshape(trans_matrix, (-1,2))
        self.transformation_matrices_right_eye["center"] = trans_matrix
    
    def adjust_left_eye_seb(self, fixations):
        
        trans_matrix = np.identity(2)
        
        corrected_fix_x = []
        corrected_fix_y = []
        
        dist_between_center_and_other_target_points = self.euclidean_distance(self.target_center, self.target_upper_left)
        
        for i in range(len(fixations[0,:])):
            current_fix = fixations[:,i]
            
            
            # On the line from upper left cornor to bottom right
            if current_fix[0] == current_fix[1]:
                pass
            
            # On the line from bottom left cornor to upper right
            elif current_fix[0] + current_fix[1] == 1.0:
                pass
            
            # In the bottom left
            elif current_fix[0] < current_fix[1]:
                # To the left
                if current_fix[0] + current_fix[1] < 1.0:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_bottom_left, self.target_upper_left, "bottom_left", "upper_left", self.transformation_matrices_left_eye)
                    
                # In the bottom
                else:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_bottom_left, self.target_bottom_right, "bottom_left", "bottom_right", self.transformation_matrices_left_eye)
            
            # In the upper right
            else:
                # In the top
                if current_fix[0] + current_fix[1] < 1.0:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_upper_right, self.target_upper_left, "upper_right", "upper_left", self.transformation_matrices_left_eye)
                    
                # To the right
                else:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_upper_right, self.target_bottom_right, "upper_right", "bottom_right", self.transformation_matrices_left_eye)
            
            
            new_fix = np.matmul(trans_matrix, current_fix)
            corrected_fix_x.append(new_fix[0])
            corrected_fix_y.append(new_fix[1])
        
        return np.array([corrected_fix_x, corrected_fix_y])
        
    def adjust_right_eye_seb(self, fixations):
        
        trans_matrix = np.identity(2)
        
        corrected_fix_x = []
        corrected_fix_y = []
        
        dist_between_center_and_other_target_points = self.euclidean_distance(self.target_center, self.target_upper_left)
        
        for i in range(len(fixations[0,:])):
            current_fix = fixations[:,i]
            
            
            # On the line from upper left cornor to bottom right
            if current_fix[0] == current_fix[1]:
                pass
            
            # On the line from bottom left cornor to upper right
            elif current_fix[0] + current_fix[1] == 1.0:
                pass
            
            # In the bottom left
            elif current_fix[0] < current_fix[1]:
                # To the left
                if current_fix[0] + current_fix[1] < 1.0:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_bottom_left, self.target_upper_left, "bottom_left", "upper_left", self.transformation_matrices_right_eye)
                    
                # In the bottom
                else:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_bottom_left, self.target_bottom_right, "bottom_left", "bottom_right", self.transformation_matrices_right_eye)
            
            # In the upper right
            else:
                # In the top
                if current_fix[0] + current_fix[1] < 1.0:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_upper_right, self.target_upper_left, "upper_right", "upper_left", self.transformation_matrices_right_eye)
                    
                # To the right
                else:
                    trans_matrix = self.setup_trans_matrix(current_fix, dist_between_center_and_other_target_points, self.target_upper_right, self.target_bottom_right, "upper_right", "bottom_right", self.transformation_matrices_right_eye)
            

            new_fix = np.matmul(trans_matrix, current_fix)
            corrected_fix_x.append(new_fix[0])
            corrected_fix_y.append(new_fix[1])
        
        return np.array([corrected_fix_x, corrected_fix_y])
    
    
    def setup_trans_matrix(self, fixation, dist_between_center_and_other_target_points, target_1, target_2, mat_1, mat_2, transformation_matrices):
        
        r_center = self.euclidean_distance(self.target_center, fixation)
        r_target_1 = self.euclidean_distance(target_1, fixation)
        r_target_2 = self.euclidean_distance(target_2, fixation)
        r_sum = r_center + r_target_1 + r_target_2
        
        
        # To enlarge the effect of a target point, if further away from the center
        rate = r_center / dist_between_center_and_other_target_points
        
                    
        rate_trans_center_temp = 1 / (r_center / r_sum)
        rate_trans_target_1_temp = 1 / (r_target_1 / r_sum)
        rate_trans_target_2_temp = 1 / (r_target_2 / r_sum)

        rate_sum = rate_trans_center_temp + rate_trans_target_1_temp + rate_trans_target_2_temp
        
        rate_trans_center = rate_trans_center_temp / rate_sum
        rate_trans_target_1 = rate_trans_target_1_temp / rate_sum
        rate_trans_target_2 = rate_trans_target_2_temp / rate_sum

        
        
#        print("Center distance")
#        print(rate_trans_center)
#        print(mat_1 + " distance")
#        print(rate_trans_target_1)
#        print(mat_2 + " distance")
#        print(rate_trans_target_2)
        
#        rate_trans_center = 0.33
#        rate_trans_target_1 = 0.33
#        rate_trans_target_2 = 0.33

#        print("Center")
#        print(transformation_matrices["center"])
#        print("Upper right")
#        print(transformation_matrices["upper_right"])
#        print("Upper left")
#        print(transformation_matrices["upper_left"])
#        print("Bottom right")
#        print(transformation_matrices["bottom_right"])
#        print("Bottom left")
#        print(transformation_matrices["bottom_left"])
        
        trans_matrix = np.identity(2)
        trans_matrix[0,0] = transformation_matrices["center"][0,0] * rate_trans_center + transformation_matrices[mat_1][0,0] * rate_trans_target_1 + transformation_matrices[mat_2][0,0] * rate_trans_target_2
        trans_matrix[0,1] = transformation_matrices["center"][0,1] * rate_trans_center + transformation_matrices[mat_1][0,1] * rate_trans_target_1 + transformation_matrices[mat_2][0,1] * rate_trans_target_2
        trans_matrix[1,0] = transformation_matrices["center"][1,0] * rate_trans_center + transformation_matrices[mat_1][1,0] * rate_trans_target_1 + transformation_matrices[mat_2][1,0] * rate_trans_target_2
        trans_matrix[1,1] = transformation_matrices["center"][1,1] * rate_trans_center + transformation_matrices[mat_1][1,1] * rate_trans_target_1 + transformation_matrices[mat_2][1,1] * rate_trans_target_2
        
#        print("Transformation matrix")
#        print(trans_matrix)
        
        return trans_matrix
    
    def adjust_left_eye_seb_2(self, fixations):
        
        trans_matrix = np.identity(2)
        
        corrected_fix_x = []
        corrected_fix_y = []
        
        dist_between_center_and_other_target_points = self.euclidean_distance(self.target_center, self.target_upper_left)
        
        for i in range(len(fixations[0,:])):
            current_fix = fixations[:,i]
            
            r_center = self.euclidean_distance(self.target_center, current_fix)
            r_upper_left = self.euclidean_distance(self.target_upper_left, current_fix)
            r_upper_right = self.euclidean_distance(self.target_upper_right, current_fix)
            r_bottom_left= self.euclidean_distance(self.target_bottom_left, current_fix)
            r_bottom_right = self.euclidean_distance(self.target_bottom_right, current_fix)
            r_sum = r_center + r_upper_left + r_upper_right + r_bottom_left + r_bottom_right
            
            
            # To enlarge the effect of a target point, if further away from the center
            rate = r_center / dist_between_center_and_other_target_points
            
                        
            rate_trans_center_temp = 1 / (r_center / r_sum)
            rate_trans_upper_left_temp = 1 / (r_upper_left / r_sum)
            rate_trans_upper_right_temp = 1 / (r_upper_right / r_sum)
            rate_trans_bottom_left_temp = 1 / (r_bottom_left / r_sum)
            rate_trans_bottom_right_temp = 1 / (r_bottom_right / r_sum)
    
            rate_sum = rate_trans_center_temp + rate_trans_upper_left_temp + rate_trans_upper_right_temp + rate_trans_bottom_left_temp + rate_trans_bottom_right_temp
            
            rate_trans_center = rate_trans_center_temp / rate_sum
            rate_trans_upper_left = rate_trans_upper_left_temp / rate_sum
            rate_trans_upper_right = rate_trans_upper_right_temp / rate_sum
            rate_trans_bottom_left = rate_trans_bottom_left_temp / rate_sum
            rate_trans_bottom_right = rate_trans_bottom_right_temp / rate_sum
                
            trans_matrix = np.identity(2)
            trans_matrix[0,0] = self.transformation_matrices_left_eye["center"][0,0] * rate_trans_center + self.transformation_matrices_left_eye["upper_left"][0,0] * rate_trans_upper_left + self.transformation_matrices_left_eye["upper_right"][0,0] * rate_trans_upper_right + self.transformation_matrices_left_eye["bottom_left"][0,0] * rate_trans_bottom_left + self.transformation_matrices_left_eye["bottom_right"][0,0] * rate_trans_bottom_right
            trans_matrix[0,1] = self.transformation_matrices_left_eye["center"][0,1] * rate_trans_center + self.transformation_matrices_left_eye["upper_left"][0,1] * rate_trans_upper_left + self.transformation_matrices_left_eye["upper_right"][0,1] * rate_trans_upper_right + self.transformation_matrices_left_eye["bottom_left"][0,1] * rate_trans_bottom_left + self.transformation_matrices_left_eye["bottom_right"][0,1] * rate_trans_bottom_right
            trans_matrix[1,0] = self.transformation_matrices_left_eye["center"][1,0] * rate_trans_center + self.transformation_matrices_left_eye["upper_left"][1,0] * rate_trans_upper_left + self.transformation_matrices_left_eye["upper_right"][1,0] * rate_trans_upper_right + self.transformation_matrices_left_eye["bottom_left"][1,0] * rate_trans_bottom_left + self.transformation_matrices_left_eye["bottom_right"][1,0] * rate_trans_bottom_right
            trans_matrix[1,1] = self.transformation_matrices_left_eye["center"][1,1] * rate_trans_center + self.transformation_matrices_left_eye["upper_left"][1,1] * rate_trans_upper_left + self.transformation_matrices_left_eye["upper_right"][1,1] * rate_trans_upper_right + self.transformation_matrices_left_eye["bottom_left"][1,1] * rate_trans_bottom_left + self.transformation_matrices_left_eye["bottom_right"][1,1] * rate_trans_bottom_right
        
            new_fix = np.matmul(trans_matrix, current_fix)
            corrected_fix_x.append(new_fix[0])
            corrected_fix_y.append(new_fix[1])
        
        return np.array([corrected_fix_x, corrected_fix_y])
    
    def adjust_right_eye_seb_2(self, fixations):
        
        trans_matrix = np.identity(2)
        
        corrected_fix_x = []
        corrected_fix_y = []
        
        dist_between_center_and_other_target_points = self.euclidean_distance(self.target_center, self.target_upper_left)
        
        for i in range(len(fixations[0,:])):
            current_fix = fixations[:,i]
            
            r_center = self.euclidean_distance(self.target_center, current_fix)
            r_upper_left = self.euclidean_distance(self.target_upper_left, current_fix)
            r_upper_right = self.euclidean_distance(self.target_upper_right, current_fix)
            r_bottom_left= self.euclidean_distance(self.target_bottom_left, current_fix)
            r_bottom_right = self.euclidean_distance(self.target_bottom_right, current_fix)
            r_sum = r_center + r_upper_left + r_upper_right + r_bottom_left + r_bottom_right
            
            
            # To enlarge the effect of a target point, if further away from the center
            rate = r_center / dist_between_center_and_other_target_points
            
                        
            rate_trans_center_temp = 1 / (r_center / r_sum)
            rate_trans_upper_left_temp = 1 / (r_upper_left / r_sum)
            rate_trans_upper_right_temp = 1 / (r_upper_right / r_sum)
            rate_trans_bottom_left_temp = 1 / (r_bottom_left / r_sum)
            rate_trans_bottom_right_temp = 1 / (r_bottom_right / r_sum)
    
            rate_sum = rate_trans_center_temp + rate_trans_upper_left_temp + rate_trans_upper_right_temp + rate_trans_bottom_left_temp + rate_trans_bottom_right_temp
            
            rate_trans_center = rate_trans_center_temp / rate_sum
            rate_trans_upper_left = rate_trans_upper_left_temp / rate_sum
            rate_trans_upper_right = rate_trans_upper_right_temp / rate_sum
            rate_trans_bottom_left = rate_trans_bottom_left_temp / rate_sum
            rate_trans_bottom_right = rate_trans_bottom_right_temp / rate_sum
                
            trans_matrix = np.identity(2)
            trans_matrix[0,0] = self.transformation_matrices_right_eye["center"][0,0] * rate_trans_center + self.transformation_matrices_right_eye["upper_left"][0,0] * rate_trans_upper_left + self.transformation_matrices_right_eye["upper_right"][0,0] * rate_trans_upper_right + self.transformation_matrices_right_eye["bottom_left"][0,0] * rate_trans_bottom_left + self.transformation_matrices_right_eye["bottom_right"][0,0] * rate_trans_bottom_right
            trans_matrix[0,1] = self.transformation_matrices_right_eye["center"][0,1] * rate_trans_center + self.transformation_matrices_right_eye["upper_left"][0,1] * rate_trans_upper_left + self.transformation_matrices_right_eye["upper_right"][0,1] * rate_trans_upper_right + self.transformation_matrices_right_eye["bottom_left"][0,1] * rate_trans_bottom_left + self.transformation_matrices_right_eye["bottom_right"][0,1] * rate_trans_bottom_right
            trans_matrix[1,0] = self.transformation_matrices_right_eye["center"][1,0] * rate_trans_center + self.transformation_matrices_right_eye["upper_left"][1,0] * rate_trans_upper_left + self.transformation_matrices_right_eye["upper_right"][1,0] * rate_trans_upper_right + self.transformation_matrices_right_eye["bottom_left"][1,0] * rate_trans_bottom_left + self.transformation_matrices_right_eye["bottom_right"][1,0] * rate_trans_bottom_right
            trans_matrix[1,1] = self.transformation_matrices_right_eye["center"][1,1] * rate_trans_center + self.transformation_matrices_right_eye["upper_left"][1,1] * rate_trans_upper_left + self.transformation_matrices_right_eye["upper_right"][1,1] * rate_trans_upper_right + self.transformation_matrices_right_eye["bottom_left"][1,1] * rate_trans_bottom_left + self.transformation_matrices_right_eye["bottom_right"][1,1] * rate_trans_bottom_right
        
            new_fix = np.matmul(trans_matrix, current_fix)
            corrected_fix_x.append(new_fix[0])
            corrected_fix_y.append(new_fix[1])
        
        return np.array([corrected_fix_x, corrected_fix_y])
    
    
    def euclidean_distance(self, q, p):
        return ((q[0]-p[0])**2+(q[1]-p[1])**2)**0.5
    
    def seperate_fixations(self, fixations):
        fixation_center = []
        fixation_center.append([])
        fixation_center.append([])
        
        fixation_upper_left = []
        fixation_upper_left.append([])
        fixation_upper_left.append([])
        
        fixation_bottom_left = []
        fixation_bottom_left.append([])
        fixation_bottom_left.append([])
        
        fixation_upper_right= []
        fixation_upper_right.append([])
        fixation_upper_right.append([])

        fixation_bottom_right = []
        fixation_bottom_right.append([])
        fixation_bottom_right.append([])
        
        target_points_center = []
        target_points_center.append([])
        target_points_center.append([])

        target_points_upper_left = []
        target_points_upper_left.append([])
        target_points_upper_left.append([])
        
        target_points_bottom_left = []
        target_points_bottom_left.append([])
        target_points_bottom_left.append([])
        
        target_points_upper_right= []
        target_points_upper_right.append([])
        target_points_upper_right.append([])

        target_points_bottom_right = []
        target_points_bottom_right.append([])
        target_points_bottom_right.append([])
        
        for i in range(len(fixations[0,:])):
            current_target = self.targets[:,i]
            
            # If upper right cornor
            if current_target[0] > 0.5 and current_target[1] < 0.5:
                fixation_upper_right[0].append(fixations[0,i])
                fixation_upper_right[1].append(fixations[1,i])
                target_points_upper_right[0].append(self.targets[0,i])
                target_points_upper_right[1].append(self.targets[1,i])
                
                self.target_upper_right = current_target
                
            # Else if upper left cornor
            elif current_target[0] < 0.5 and current_target[1] < 0.5:
                fixation_upper_left[0].append(fixations[0,i])
                fixation_upper_left[1].append(fixations[1,i])
                target_points_upper_left[0].append(self.targets[0,i])
                target_points_upper_left[1].append(self.targets[1,i])
                
                self.target_upper_left = current_target
                
            # Else if bottom right cornor
            elif current_target[0] > 0.5 and current_target[1] > 0.5:
                fixation_bottom_right[0].append(fixations[0,i])
                fixation_bottom_right[1].append(fixations[1,i])
                target_points_bottom_right[0].append(self.targets[0,i])
                target_points_bottom_right[1].append(self.targets[1,i])

                self.target_bottom_right= current_target

            # Else if bottom left cornor
            elif current_target[0] < 0.5 and current_target[1] > 0.5:
                fixation_bottom_left[0].append(fixations[0,i])
                fixation_bottom_left[1].append(fixations[1,i])
                target_points_bottom_left[0].append(self.targets[0,i])
                target_points_bottom_left[1].append(self.targets[1,i])

                self.target_bottom_left = current_target

            # Else if center
            else:
                fixation_center[0].append(fixations[0,i])
                fixation_center[1].append(fixations[1,i])
                target_points_center[0].append(self.targets[0,i])
                target_points_center[1].append(self.targets[1,i])
                
                self.target_center = current_target
                
        return (np.array(fixation_upper_right), np.array(fixation_upper_left), np.array(fixation_bottom_right), np.array(fixation_bottom_left), np.array(fixation_center), np.array(target_points_upper_right), np.array(target_points_upper_left), np.array(target_points_bottom_right), np.array(target_points_bottom_left), np.array(target_points_center))
    
    
    def calibrate_eyes_regression(self, gaze_data_left, gaze_data_right):
        
        degree = 2
        
        print("Calibrating eyes regression\n----------------")
        target_points = self.targets
        # the gaze data recorded is normalized
        # flip y-coordinates to turn recording coordinate system (origo in top-left) into screen coordinate system (origo in bottom-left)
#        px_left_x = gaze_data_left[0,:] * self.px_width
#        px_left_y = self.px_height - gaze_data_left[1,:] * self.px_height
#        px_right_x = gaze_data_right[0,:] * self.px_width
#        px_right_y = self.px_height - gaze_data_right[1,:] * self.px_height
        
        px_left_x = gaze_data_left[0,:]
        px_left_y = gaze_data_left[1,:]
        px_right_x = gaze_data_left[0,:]
        px_right_y = gaze_data_left[1,:]
        
        
                
#        px_target_x = target_points[0,:] * self.px_width
#        px_target_y = self.screen_height_px - tar get_points[1,:] * self.px_height
        
        
        
        ## CAlCULATE VERTICAL ERRORS AS GAZE VARIES HORIZONTALLY
        # convert normalized coordinates to pixel coordinates (as on screen)
        pixel_err_left = [(fix_x-tar_x, fix_y-tar_y) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], target_points[0,:], target_points[1,:])]
        pixel_err_right = [(fix_x-tar_x, fix_y-tar_y) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:])]
        
        px_err_left_x = []
        px_err_left_y = []
        px_err_right_x = []
        px_err_right_y = []
        for err_left_norm, err_right_norm in zip(pixel_err_left, pixel_err_right):
#            px_err_left_x.append(err_left_norm[0] * self.px_width)
#            px_err_left_y.append(err_left_norm[1] * self.px_height)
#            px_err_right_x.append(err_right_norm[0] * self.px_width)
#            px_err_right_y.append(err_right_norm[1] * self.px_height) 
            
            px_err_left_x.append(err_left_norm[0])
            px_err_left_y.append(err_left_norm[1])
            px_err_right_x.append(err_right_norm[0])
            px_err_right_y.append(err_right_norm[1]) 
            
        # remove outliers
#        px_err_left_x, px_left_y = self.reject_outliers(px_err_left_x, px_left_y)
#        px_err_left_y, px_left_x = self.reject_outliers(px_err_left_y, px_left_x)
#        px_err_right_x, px_right_y = self.reject_outliers(px_err_right_x, px_right_y)
#        px_err_right_y, px_right_x = self.reject_outliers(px_err_right_y, px_right_x)
        
#        print(px_err_left_x)
        
        # fit a qudratic line for the vertical errors
        self.poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, degree))
        self.poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_x, degree))
        self.poly_right_x = np.poly1d(np.polyfit(px_right_x, px_err_right_y, degree))
        self.poly_right_y = np.poly1d(np.polyfit(px_right_y, px_err_right_x, degree))
        
        
    def reject_outliers(self, data, targets, m=1.5):
        prep_data = [a if abs(a - np.mean(data)) < m * np.std(data) else sys.maxint for a in data]

        filtered_data = []
        filtered_targets = []
        for a, b in zip(prep_data, targets):
            if a != sys.maxint:
                filtered_data.append(a)
                filtered_targets.append(b)
                
        return (filtered_data, filtered_targets)
    
        
     
    def adjust_left_eye_regression(self, fixations):
#        gaze_data = self.norm_to_pixels(fixations)
        gaze_data = fixations
        corrected_x = []
        corrected_y = []
        for i in range(len(fixations[0,:])):
            predicted_error_x = self.poly_left_y(gaze_data[1,i])
            predicted_error_y = self.poly_left_x(gaze_data[0,i])
            
            corrected_x.append(gaze_data[0,i] + predicted_error_x)
            corrected_y.append(gaze_data[1,i] + predicted_error_y)
            
#        return self.pixels_to_norm(np.array([corrected_x, corrected_y]))
        return np.array([corrected_x, corrected_y])
     
    def adjust_right_eye_regression(self, fixations):
#        gaze_data = self.norm_to_pixels(fixations)
        gaze_data = fixations
        corrected_x = []
        corrected_y = []
        for i in range(len(fixations[0,:])):
            predicted_error_x = self.poly_right_y(gaze_data[1,i])
            predicted_error_y = self.poly_right_x(gaze_data[0,i])
            
            corrected_x.append(gaze_data[0,i] + predicted_error_x)
            corrected_y.append(gaze_data[1,i] + predicted_error_y)
            
        
        
#        return self.pixels_to_norm(np.array([corrected_x, corrected_y]))
        return np.array([corrected_x, corrected_y])
    
    
    def norm_to_pixels(self, data):
        # flip y-coordinates to turn recording coordinate system (origo in top-left) into screen coordinate system (origo in bottom-left)
        px_x = data[0,:] * self.px_width
        px_y = self.px_height - data[1,:] * self.px_height
        
        return np.array([px_x, px_y])
        
    
    def pixels_to_norm(self, data):
        # flip y-coordinates to turn recording coordinate system (origo in top-left) into screen coordinate system (origo in bottom-left)
        norm_x = data[0,:] / self.px_width
        norm_y = 1 - data[1,:] / self.px_height
        
        return np.array([norm_x, norm_y])
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    