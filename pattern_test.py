# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:18:54 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import matplotlib.pyplot as plt
import math
import shape_similarity as ssim



# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"

# Session to run
session_folder = "ctrl_group_3_seb"
#session_folder = "infant3_d_marley_7m_2"
#session_folder = "infant2_525d_noel_6m"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "training_fixation.csv"

analyzer = gda.GazeDataAnalyzer()

print("\nSETUP TRANSFORMATION")
analyzer.setup(config_filename, cal_filename, "dbscan_fixation")



## LINEAR
##--------------------
print("\nTEST DATA - PURSUIT (LINEAR)")
training_filename = test_folder + "training_pursuit_linear.csv"

targets, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze(training_filename, None, False)
gaze_data_avg = analyzer.get_avg(gaze_data_left, gaze_data_right)
eq_targets = analyzer.get_pattern_eq("linear", targets.T)
print(eq_targets)
seg_count = 0
gaze_segments = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
colors_compliment = ['#ff8228', '#0e8fff', '#ff46fc']
        
linear_fit_gaze = []
linear_fit_target = []

gaze_data_avg = analyzer.get_avg(gaze_data_left, gaze_data_right)
for i in range(len(eq_targets)):
    startIndex = eq_targets[i][0]
    endIndex = len(gaze_data_avg[0]) if i == len(eq_targets) - 1 else eq_targets[i+1][0]
    
    linear_fit_target.append((eq_targets[i][1], eq_targets[i][2]))
    
    segment = []
    for index in range(startIndex, endIndex):
        segment.append((gaze_data_avg[0][index], gaze_data_avg[1][index]))
    gaze_segments.append(segment)
        

x_targets = targets[0,:]
y_targets = targets[1,:]
plt.plot(x_targets, y_targets, color='black')
for i in range(len(gaze_segments)):
    fix_x = [fix[0] for fix in gaze_segments[i]]
    fix_y = [fix[1] for fix in gaze_segments[i]]
    a, b = analyzer.best_fit(fix_x, fix_y)
    linear_fit_gaze.append((a,b))
    plt.scatter(fix_x, fix_y, color=colors[i])
    padding = 0.01 if i < len(gaze_segments)-1 else -0.01
    for j in range(10):
        fix_x.insert(0, fix_x[0]-padding)
        fix_x.append(fix_x[-1]+padding)
    yfit = [a*x+b for x in fix_x]   
    print('best fit line: y = {:.2f}x + {:.2f}'.format(a, b))
    
    plt.xlim(0,1)
    plt.ylim(1,0)
    plt.gca().xaxis.tick_top()
    plt.plot(fix_x, yfit, color=colors[i])
#    plt.show()
plt.show()    

# compare linear segments of target and gaze
print("target coeffs")
print(linear_fit_target)

print("gaze coeffs")
print(linear_fit_gaze)

print("")
# slope comparison
for i in range(len(linear_fit_target)):
    slope_diff = (linear_fit_target[i][0] - linear_fit_gaze[i][0]) / linear_fit_target[i][0]
    print("slope diff = " + str(abs(linear_fit_target[i][0] - linear_fit_gaze[i][0])))
    
    print(math.atan((linear_fit_gaze[i][0]-linear_fit_target[i][0])/(1+linear_fit_gaze[i][0]*linear_fit_target[i][0])))
    

plt.xlim(0,1)
plt.ylim(1,0)
plt.gca().xaxis.tick_top()

tobii_pos = [(0.25, 0.75), (0.65, 0.25), (0.75, 0.75), (0.5, 0.5)]
count = 1
for pos in tobii_pos:
    plt.scatter(pos[0], pos[1], marker="x")
    plt.text(pos[0]+0.03, pos[1]+0.03, str(count) + ": " + str(pos), fontsize="10")
    count += 1
plt.show()



### CIRCLUAR
##--------------------

print("\nTEST DATA - PURSUIT (LINEAR)")
training_filename = test_folder + "training_pursuit_circle.csv"

targets, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze(training_filename, "dbscan_pursuit")
gaze_data_avg = analyzer.get_avg(gaze_data_left, gaze_data_right)


fig = plt.figure()
frame1 = fig.add_axes((0,.4,1,.8))
#ax = fig.gca()


x_targets = targets[0,:]
y_targets = targets[1,:]
#plt.plot(x_targets, y_targets, color='black')
gaze_plot = plt.scatter(gaze_data_avg[0,:], gaze_data_avg[1,:])


circle_params_gaze = analyzer.get_pattern_eq("circle", gaze_data_avg.T)

center_pos = circle_params_gaze[0][0]
radius = circle_params_gaze[0][1]

gaze_circle = plt.Circle(center_pos, radius, fill=False, color='orange', linewidth=3)

frame1.add_artist(gaze_circle)
print("Fit circle eq:\t\t (x-" + str(round(center_pos[0],4)) + ")^2 + (y-" + str(round(center_pos[1],4)) +")^2 = " + str(round(radius,4)) + "^2")





#def circle_f(x, c, r):
#    a = 1
#    b = -2*c[1]
#    c = -r**2+(x-c[0])**2-c[1]**2
#    sol_p = (-b+(b**2-4*a*c)**0.5)/(2*a)
#    sol_n = (-b-(b**2-4*a*c)**0.5)/(2*a) 
#    return sol_p if abs(sol_p) < abs(sol_n) else sol_n
#difference = [circle_f(x, center_pos, radius)-y for x,y in zip(gaze_data_avg[0,:], gaze_data_avg[1,:])]
dist_to_circle = [((x - center_pos[0])**2 + (y-center_pos[1])**2)**0.5-radius for x,y in zip(gaze_data_avg[0,:], gaze_data_avg[1,:])]



## ---------

circle_params_target = analyzer.get_pattern_eq("circle", targets.T)

center_pos = circle_params_target[0][0]
radius = circle_params_target[0][1]

target_circle = plt.Circle(center_pos, radius, fill=False, color='black', linewidth=1)

#ax.add_artist(target_circle)
print("Target circle eq:\t (x-" + str(round(center_pos[0],4)) + ")^2 + (y-" + str(round(center_pos[1],4)) +")^2 = " + str(round(radius,4)) + "^2")


#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
#ax.legend((gaze_plot, gaze_circle, target_circle), ("gaze data avg\n(left/right eye)", "fitted circle", "target path"))
#ax.legend((gaze_plot, target_circle), ("gaze data avg\n(left/right eye)", "target path"))
#frame1.legend((gaze_plot, gaze_circle), ("gaze data avg\n(left/right eye)", "fitted circle"))
#plt.show()

frame1.set_ylim(1,0)
frame1.set_xlim(0,1)
frame1.xaxis.tick_top()
frame1.xaxis.set_label_position("top")
frame1.set_xlabel("ADCS (X)")
frame1.set_ylabel("ADCS (Y)")

frame2 = fig.add_axes((0,0,1,.4))
from matplotlib.ticker import MaxNLocator
frame2.yaxis.set_major_locator(MaxNLocator(symmetric=True))
frame2.margins(y=0.1)
#frame2.set_ylim(ymax=.08)
#import numpy as np
#frame2.set_yticks(np.arange(-0.1,0.06, step=0.1))
plt.plot(range(len(gaze_data_avg[0,:])), [0]*len(gaze_data_avg[0,:]), linestyle="--", color="black")
plt.scatter(range(len(gaze_data_avg[0,:])), dist_to_circle, marker=".", color="red")
frame2.grid()
frame2.set_xlabel("Observations")
frame2.set_ylabel("Residual distance")
plt.show()

        

    


import numpy as np
#theta = np.linspace(0, 2*np.pi, 8)
#
#c = circle_params_target[0][0]
#r = circle_params_target[0][1]
#x = c[0] + r*np.cos(theta)
#y = c[1] + r*np.sin(theta)
#plt.plot(x,y)
#
#vectors_target = []
#for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#    vectors_target.append((x2-x1, y2-y1))
#    
#slopes_target = []
#for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#    if x1 == x2:
#        slopes_target.append(float(2**31-1))
#        continue
#    slopes_target.append((y2-y1)/(x2-x1))
#
#plt.ylim(1,0)
#plt.xlim(0,1)
#plt.gca().xaxis.tick_top()
#plt.show()
#
#vectors_gaze = []
#for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#    vectors_gaze.append((x2-x1, y2-y1))
#
#slopes_gaze = []
#for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#    if x1 == x2:
#        slopes_gaze.append(float(2**31-1))
#        continue
#    slopes_gaze.append((y2-y1)/(x2-x1))
    

#print(slopes_target)
#print(slopes_gaze)
#
#print("")
#slopes_diff = [t-g for t,g in zip(slopes_target, slopes_gaze)]
#slope_comparison = [math.atan((t-g)/(1+g*t)) for t,g in zip(slopes_target, slopes_gaze)]
#
##print(slope_comparison)
#print("Slope similarity measure: \t" + str(1-np.array(slope_comparison).mean()))



N = 3



gaze_x = np.append(gaze_data_avg[0,:], gaze_data_avg[0,0])
gaze_y = np.append(gaze_data_avg[1,:], gaze_data_avg[1,0])
########################
##############################

vectors_gaze_avg = []
for x1, x2, y1, y2 in zip(gaze_x, gaze_x[1:], gaze_y, gaze_y[1:]):
    v = (x2-x1, y2-y1)
    if v == (0.0, 0.0):
        continue
    vectors_gaze_avg.append((x2-x1, y2-y1))
    
    
vectors_gaze_avg = []
for i in range(N):  
    v = (0,0)
    for j in range(i*len(gaze_x)/N, i*len(gaze_x)/N + len(gaze_x)/N):
        v0 = (gaze_x[j], gaze_y[j])
        v = ssim.add(v, v0)
    vectors_gaze_avg.append(v)
    
        
    
slopes_gaze_avg = []
for x1, x2, y1, y2 in zip(gaze_x, gaze_x[1:], gaze_y, gaze_y[1:]):
    if x1 == x2:
        slopes_gaze_avg.append(float(2**31-1))
        continue
    slopes_gaze_avg.append((y2-y1)/(x2-x1))
    

slopes_gaze_avg_multiple = []
for i in range(N):
    
    
    body = [[],[]]
    for j in range(i*len(gaze_x)/N, i*len(gaze_x)/N + len(gaze_x)/N):
        body[0].append(gaze_data_avg[0,(i-j) % len(gaze_data_avg[0,:])])
        body[1].append(gaze_data_avg[1,(i-j) % len(gaze_data_avg[1,:])])
        
#        print(j)
        
#    for j in range(i*len(gaze_x)/N, i*len(gaze_x)/N + len(gaze_x)/N):    
#        body[0].append(gaze_data_avg[0,(i+(j+1)) % len(gaze_data_avg[0,:])])
#        body[1].append(gaze_data_avg[1,(i+(j+1)) % len(gaze_data_avg[1,:])])
    
    body = np.array(body)
    
    slopes = []
    for x1, x2, y1, y2 in zip(body[0,:], body[0,:][1:], body[1,:], body[1,:][1:]):
        if x1 == x2:
            slopes.append(float(2**31-1))
            continue
        slopes.append((y2-y1)/(x2-x1))
    
    slopes_gaze_avg_multiple.append(np.array(slopes).mean())


##########################
###############################

    
circle_params_target = analyzer.get_pattern_eq("circle", targets.T)

theta = np.linspace(0, 2*np.pi, N+1)

c = circle_params_target[0][0]
r = circle_params_target[0][1]
target_x = c[0] + r*-np.cos(theta)
target_y = c[1] + r*np.sin(theta)


vectors_target = []
for x1, x2, y1, y2 in zip(target_x, target_x[1:], target_y, target_y[1:]):
    vectors_target.append((x2-x1, y2-y1))
    
    
slopes_target = []
for x1, x2, y1, y2 in zip(target_x, target_x[1:], target_y, target_y[1:]):
    if x1 == x2:
        slopes_target.append(float(2**31-1))
        continue
    slopes_target.append((y2-y1)/(x2-x1))
    
    
######################
##########################
    
    
    
print("")
slopes_diff = [t-g for t,g in zip(slopes_target, slopes_gaze_avg)]

print(slopes_target)
print(slopes_gaze_avg_multiple)

slope_comparison = [math.atan(math.tan(abs(math.atan(t)-math.atan(g)))/(1+math.tan(abs(math.atan(g)*math.atan(t))))) for t,g in zip(slopes_target, slopes_gaze_avg_multiple)]

print(slope_comparison)

#print(slopes_target)
#print(slopes_gaze_avg)

angle_comparison = [2*abs(ssim.angle_between(v1,v2))/np.pi for v1, v2 in zip(vectors_target, vectors_gaze_avg)]
print(vectors_gaze_avg)
print(vectors_target)


#print(angle_comparison)
#print(slope_comparison)
plt.ylim(1,0)
plt.xlim(0,1)
plt.gca().xaxis.tick_top()
plt.plot(gaze_x, gaze_y)
plt.plot(target_x, target_y)

plt.show()
#print(slope_comparison)
print("Slope similarity measure: \t" + str(1-np.array(slope_comparison).mean()))
#print("Slope similarity measure diff: \t" + str(1-np.array(slope_diff).mean()))
print("Angle similarity measure: \t" + str(1-np.array(angle_comparison).mean()))




#for i in range(len(vectors_gaze_avg)):
#    theta = np.linspace(0, 2*np.pi, i+1)
#    
#    c = circle_params_target[0][0]
#    r = circle_params_target[0][1]
#    x = c[0] + r*np.cos(theta)
#    y = c[1] + r*np.sin(theta)
#    
#    vectors_target = []
#    for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#        vectors_target.append((x2-x1, y2-y1))
#    
#    print(str(i+1) + ": " + str(1-(np.array([ssim.angle_between(v1,v2)/np.pi for v1, v2 in zip(vectors_target, vectors_gaze_avg)]).mean())))
    

























#import shape_similarity as ssim
## Sim measure between circle of targets and polygon contour shape gaze data
#print("Shape similary measure (pdf): \t" + str(ssim.sim_measure(vectors_target, vectors_gaze_avg)))
#
#triangle = np.array([[0.5, 0.3333, 0.6666, 0.5], [0.3333, 0.6666, 0.6666, 0.3333]])
#vectors_triangle = []
#for x1, x2, y1, y2 in zip(triangle[0,:], triangle[0,:][1:], triangle[1,:], triangle[1,:][1:]):
#    v = (x2-x1, y2-y1)
#    if v == (0.0, 0.0):
#        continue
#    vectors_triangle.append(ssim.unit_vector((x2-x1, y2-y1)))  
#
#print("Shape similary measure (pdf): Triangle vs. Target-circle: \t" + str(ssim.sim_measure(vectors_target, vectors_triangle)))
#
#
#gaze_x = np.append(gaze_data_avg[0,:], gaze_data_avg[0,0])
#gaze_y = np.append(gaze_data_avg[1,:], gaze_data_avg[1,0])
#
##plt.ylim(1,0)
##plt.xlim(0,1)
##plt.gca().xaxis.tick_top()
##plt.plot(gaze_x, gaze_y)
##plt.show()
#
#gaze_points = [(gx,gy) for gx,gy in zip(gaze_x, gaze_y)]
#import rdp
#
#plt.ylim(1,0)
#plt.xlim(0,1)
#plt.gca().xaxis.tick_top()
#new_points = rdp.simplifyDouglasPeucker(gaze_points, 35) 
#plt.plot([p[0] for p in new_points], [p[1] for p in new_points])
#
#gaze_rdp = np.array([[p[0] for p in new_points], [p[1] for p in new_points]])
#
#rdpVectors = []
#for x1, x2, y1, y2 in zip(gaze_rdp[0,:], gaze_rdp[0,:][1:], gaze_rdp[1,:], gaze_rdp[1,:][1:]):
#    v = (x2-x1, y2-y1)
#    if v == (0.0, 0.0):
#        continue
#    rdpVectors.append(ssim.unit_vector((x2-x1, y2-y1)))   
#
#
#print("Shape similary measure (pdf): Gaze RDP vs. Target-circle: \t" + str(ssim.sim_measure(vectors_target, rdpVectors)))

#bestSimN = 0
#bestSim = -10000
#for i in range(30):
#    theta = np.linspace(0, 2*np.pi, i+5)
#    
#    c = circle_params_target[0][0]
#    r = circle_params_target[0][1]
#    x = c[0] + r*np.cos(theta)
#    y = c[1] + r*np.sin(theta)
#    plt.plot(x,y)
#    
#    vectors_target = []
#    for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
#        vectors_target.append(ssim.unit_vector((x2-x1, y2-y1)))
#    
#    if ssim.sim_measure(vectors_target, rdpVectors) > bestSim:
#        bestSim = ssim.sim_measure(vectors_target, rdpVectors)
#        bestSimN = i+5
#    
#print(bestSim)
#print(bestSimN)



