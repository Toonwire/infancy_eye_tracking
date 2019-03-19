#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 08:27:12 2018

@author: sebastiannyholm
"""

class DBScan:

    def __init__(self):
        pass
    
    def euclidean_distance(self, q, p):
        return ((q[0]-p[0])**2+(q[1]-p[1])**2)**0.5
    
    # Scan all points 
    # Compute distance and check eps
    # Add to result
    def range_query(self, points, p, eps):
        return [q for q in points if p != q and self.euclidean_distance(p,q) <= eps]
        
    ## labels
    #-1 -> undefined
    # 0 -> noise
    #1+ -> clusters
    #####
    def run(self, points, eps, minPts):
        clusterCount = 0
        labels = {}
        
        points = [(p[0], p[1]) for p in points]                     # convert to tuples
        
        for p in points:
            if p in labels:
                continue
            
            neighbors = self.range_query(points, p, eps)            # Find neighbors
            
            if len(neighbors) < minPts:                             # Density check
                labels[p] = 0                                       # Label as Noise
                continue
            
            clusterCount = clusterCount + 1
            labels[p] = clusterCount
            
            neighborsToExpand = neighbors
            
            i = 0
            while i < len(neighborsToExpand):
                q = neighborsToExpand[i]
                i = i+1
                    
                if q in labels:
                    if labels[q] == 0:
                        labels[q] = clusterCount
                    
                    continue
                
                labels[q] = clusterCount
                newNeighbors = self.range_query(points, q, eps)     # Find more neighbors

                if len(newNeighbors) >= minPts:                     # Density check
                    neighborsToExpand += newNeighbors               # Add new neighbors
                    
        return labels






