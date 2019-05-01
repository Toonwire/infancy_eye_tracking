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
            
            i = 0
            while i < len(neighbors):
                q = neighbors[i]
                i = i+1
                    
                if q in labels:
                    if labels[q] == 0:
                        labels[q] = clusterCount
                    
                    continue
                
                labels[q] = clusterCount
                newNeighbors = self.range_query(points, q, eps)     # Find more neighbors

                if len(newNeighbors) >= minPts:                     # Density check
                    neighbors += newNeighbors               # Add new neighbors
                    
        return labels



    # Scan all points 
    # Compute distance and check eps
    # Add to result
    def range_query_linear(self, points, start_index, p, eps):
        
        neighbors = []
        
        misses = 0
        for i in range(start_index + 1, len(points)):
            
            q = points[i]
            
            if self.euclidean_distance(p,q) <= eps:
                neighbors.append(q)
                misses = 0
            else:
                misses += 1

            if misses == 3:
                break
            
        return neighbors

    ## labels
    #-1 -> undefined
    # 0 -> noise
    #1+ -> clusters
    #####
    def run_linear(self, points, eps, minPts):
        clusterCount = 0
        labels = {}
        
        points = [(p[0], p[1]) for p in points]                     # convert to tuples
        
        for i, p in enumerate(points):
            
            if p in labels:
                continue
            
            neighbors = self.range_query_linear(points, i, p, eps)     # Find neighbors
            
            j = 0
            while j < len(neighbors):
                q = neighbors[j]
                j += 1
                
                newNeighbors = self.range_query_linear(points, i+j, q, eps)     # Find more neighbors
                
                for n in newNeighbors:
                    if n not in neighbors:
                        neighbors.append(n)                                      # Add new neighbors

            
            if len(neighbors) < minPts:                             # Density check
                labels[p] = 0                                       # Label as Noise
                continue
            
            clusterCount = clusterCount + 1
            labels[p] = clusterCount
            
            for q in neighbors:
                labels[q] = clusterCount
            
            
        return labels




