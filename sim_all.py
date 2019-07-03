# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import numpy as np

def analyze(session_folder):
    
    try:
        print("Running for " + session_folder)
        print("------------------------")
        
        # Setting path and files
        session_path = "session_data/" + session_folder + "/"
        test_folder = session_path + "test_" + type_of_cal + "/"
        config_filename = session_path + "config.csv"
#        cal_filename = test_folder + "training_fixation.csv"
        cal_filename = test_folder + "training_fixation_2.csv"
#        cal_filename = test_folder + "training_pursuit_circle.csv"
#        cal_filename = test_folder + "training_pursuit_linear.csv"
#        cal_filename = test_folder + "training_pursuit_spiral.csv"
        
#        remove_outliers = True
        remove_outliers = False

        
        print("")
        print("Computing analyze linear transformation")
        print("------------------------")
        
        analyzer = gda.GazeDataAnalyzer()
        print("\nSETUP TRANSFORMATION")
#        analyzer.cross_validation(config_filename, cal_filename, "dbscan_fixation", k = 2)
#        analyzer.cross_validation(config_filename, cal_filename, "dbscan_pursuit", k = 5)
        analyzer.setup_affine2(config_filename, cal_filename, "dbscan_fixation")
#        analyzer.setup_affine2(config_filename, cal_filename, "dbscan_pursuit")
#        print("\nTRAINING DATA")
#        analyzer.analyze(cal_filename, "dbscan_fixation")
        
        try:
            print("\nTEST DATA - FIXATION")
            training_filename = test_folder + "training_fixation.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_fixation", "values", remove_outliers = remove_outliers)
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
            fixation_deg_raw.append(np.mean(angle_avg))
            fixation_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING FIXATION")
            print("-----------------------------")
            fixation_deg_raw.append(0)
            fixation_deg_cor.append(0)
            
        try:
            print("\nTEST DATA - FIXATION_2")
            training_filename = test_folder + "training_fixation_2.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_fixation", "values", remove_outliers = remove_outliers)
        
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
            fixation_2_deg_raw.append(np.mean(angle_avg))
            fixation_2_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING FIXATION 2")
            print("-----------------------------")
            fixation_2_deg_raw.append(0)    
            fixation_2_deg_cor.append(0)
            
        try:
            print("\nTEST DATA - PURSUIT (CIRCLE)")
            training_filename = test_folder + "training_pursuit_circle.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_pursuit", "values", remove_outliers = remove_outliers)
        
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
        
            pursuit_circle_deg_raw.append(np.mean(angle_avg))
            pursuit_circle_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING PURSUIT CIRCLE")
            print("-----------------------------")
            pursuit_circle_deg_raw.append(0)
            pursuit_circle_deg_cor.append(0)
            
        try:
            print("\nTEST DATA - PURSUIT (CIRCLE REVERT)")
            training_filename = test_folder + "training_pursuit_circle_revert.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_pursuit", "values", remove_outliers = remove_outliers)
        
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
        
            pursuit_circle_revert_deg_raw.append(np.mean(angle_avg))
            pursuit_circle_revert_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING PURSUIT CIRCLE REVERT")
            print("-----------------------------")
            pursuit_circle_revert_deg_raw.append(0)
            pursuit_circle_revert_deg_cor.append(0)
            
            
        try:
            print("\nTEST DATA - PURSUIT (LINEAR)")
            training_filename = test_folder + "training_pursuit_linear.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_pursuit", "values", remove_outliers = remove_outliers)
    
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
                
            pursuit_linear_deg_raw.append(np.mean(angle_avg))
            pursuit_linear_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING PURSUIT LINEAR")
            print("-----------------------------")
            pursuit_linear_deg_raw.append(0)
            pursuit_linear_deg_cor.append(0)
            
        try:
            print("\nTEST DATA - PURSUIT (SPIRAL)")
            training_filename = test_folder + "training_pursuit_spiral.csv"
            angle_avg, angle_avg_corrected = analyzer.analyze_affine2(training_filename, "dbscan_pursuit", "values", remove_outliers = remove_outliers)
        
#            if len(angle_avg) < (5*90*0.3):
#                raise ValueError('Not enough data')
                
            pursuit_spiral_deg_raw.append(np.mean(angle_avg))
            pursuit_spiral_deg_cor.append(np.mean(angle_avg_corrected))
        except:
            print("-----------------------------")
            print("SKIPPING PURSUIT SPIRAL")
            print("-----------------------------")
            pursuit_spiral_deg_raw.append(0)
            pursuit_spiral_deg_cor.append(0)
        
        print("")
    except:
        print("-----------------------------")
        print("SKIPPING " + session_folder)
        print("-----------------------------")
    
print("Calibrating for default:")
print("------------------------")
# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"

print("")

fixation_deg_raw = []
fixation_deg_cor = []
# ------------
fixation_2_deg_raw = []
fixation_2_deg_cor = []
# ------------
pursuit_circle_deg_raw = []
pursuit_circle_deg_cor = []
# ------------
pursuit_circle_revert_deg_raw = []
pursuit_circle_revert_deg_cor = []
# ------------
pursuit_linear_deg_raw = []
pursuit_linear_deg_cor = []
# ------------
pursuit_spiral_deg_raw = []
pursuit_spiral_deg_cor = []


for i in range(1):
    # Session to run
#    analyze("ctrl_group_2_louise")
#    analyze("ctrl_group_2_lasse")
#    analyze("ctrl_group_2_marie")
#    analyze("ctrl_group_2_mikkel")
#    analyze("ctrl_group_2_lukas")
#    analyze("ctrl_group_2_seb")
    
    analyze("ctrl_group_3_seb")
    analyze("ctrl_group_3_lukas")
    
#    analyze("ctrl4_a_seb_glass")
#    analyze("ctrl4_a_seb")
#    analyze("ctrl4_a_marie_2")
#    analyze("ctrl4_a_marie")
#    analyze("ctrl4_a_lukas_blind")
#    analyze("ctrl4_a_lukas")
    
#    analyze("infant2_52d_sofie_7m")
#    analyze("infant2_525d_noel_6m")
#    analyze("infant2_d2_viggo_6m")
#    analyze("infant2_d52_vilja_7m")
#    analyze("infant3_25d_marley_7m")


def print_nice(values):
    my_str = ""
    for i in range(len(values)):
        my_str += str(values[i])

        if i < len(values) - 1:
            my_str += ", "
            
    return my_str


fixation_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(fixation_deg_raw, fixation_deg_cor)]
fixation_2_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(fixation_2_deg_raw, fixation_2_deg_cor)]
pursuit_circle_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(pursuit_circle_deg_raw, pursuit_circle_deg_cor)]
pursuit_circle_revert_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(pursuit_circle_revert_deg_raw, pursuit_circle_revert_deg_cor)]
pursuit_linear_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(pursuit_linear_deg_raw, pursuit_linear_deg_cor)]
pursuit_spiral_deg_imp = [(raw-cor)/raw*100 if raw > 0 else 0 for raw, cor in zip(pursuit_spiral_deg_raw, pursuit_spiral_deg_cor)]

print("Improvements values")
print(print_nice(fixation_deg_imp))
print(print_nice(fixation_2_deg_imp))
print(print_nice(pursuit_circle_deg_imp))
print(print_nice(pursuit_circle_revert_deg_imp))
print(print_nice(pursuit_linear_deg_imp))
print(print_nice(pursuit_spiral_deg_imp))


print("Correction values")
print(print_nice(fixation_deg_cor))
print(print_nice(fixation_2_deg_cor))
print(print_nice(pursuit_circle_deg_cor))
print(print_nice(pursuit_circle_revert_deg_cor))
print(print_nice(pursuit_linear_deg_cor))
print(print_nice(pursuit_spiral_deg_cor)) 
   

print("Raw values")
print(print_nice(fixation_deg_raw))
print(print_nice(fixation_2_deg_raw))
print(print_nice(pursuit_circle_deg_raw))
print(print_nice(pursuit_circle_revert_deg_raw))
print(print_nice(pursuit_linear_deg_raw))
print(print_nice(pursuit_spiral_deg_raw))



print("")
print("Fixations value")
print("Average RMSE degree after correction: " + str(np.mean(fixation_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(fixation_deg_imp)))

print("")
print("Fixations 2 value")
print("Average RMSE degree after correction: " + str(np.mean(fixation_2_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(fixation_2_deg_imp)))

print("")
print("Pursuit circle value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_circle_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_circle_deg_imp)))

print("")
print("Pursuit circle value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_circle_revert_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_circle_revert_deg_imp)))

print("")
print("Pursuit linear value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_linear_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_linear_deg_imp)))

print("")
print("Pursuit spiral value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_spiral_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_spiral_deg_imp)))

#print("")
#print("All value")
#print("Average RMSE degree after correction: " + str((np.mean(fixation_deg_cor) + np.mean(pursuit_circle_deg_cor) + np.mean(pursuit_linear_deg_cor) + np.mean(pursuit_spiral_deg_cor)) / 4))
#print("Average RMSE improvement: " + str((np.mean(fixation_deg_imp) + np.mean(pursuit_circle_deg_imp) + np.mean(pursuit_linear_deg_imp) + np.mean(pursuit_spiral_deg_imp)) / 4))
#
#print("")
#print("Smooth pursuit value")
#print("Average RMSE degree after correction: " + str((np.mean(pursuit_circle_deg_cor) + np.mean(pursuit_linear_deg_cor) + np.mean(pursuit_spiral_deg_cor)) / 3))
#print("Average RMSE improvement: " + str((np.mean(pursuit_circle_deg_imp) + np.mean(pursuit_linear_deg_imp) + np.mean(pursuit_spiral_deg_imp)) / 3))

print("")
print("All value")
print("Average RMSE degree after correction: " + str((np.mean(fixation_deg_cor) + np.mean(fixation_2_deg_cor) + np.mean(pursuit_circle_deg_cor) + np.mean(pursuit_circle_revert_deg_cor) + np.mean(pursuit_linear_deg_cor) + np.mean(pursuit_spiral_deg_cor)) / 6))
print("Average RMSE improvement: " + str((np.mean(fixation_deg_imp) + np.mean(fixation_2_deg_imp) + np.mean(pursuit_circle_deg_imp) + np.mean(pursuit_circle_revert_deg_imp) + np.mean(pursuit_linear_deg_imp) + np.mean(pursuit_spiral_deg_imp)) / 6))

print("")
print("Smooth pursuit value")
print("Average RMSE degree after correction: " + str((np.mean(pursuit_circle_deg_cor) + np.mean(pursuit_circle_revert_deg_cor) + np.mean(pursuit_linear_deg_cor) + np.mean(pursuit_spiral_deg_cor)) / 4))
print("Average RMSE improvement: " + str((np.mean(pursuit_circle_deg_imp) + np.mean(pursuit_circle_revert_deg_imp) + np.mean(pursuit_linear_deg_imp) + np.mean(pursuit_spiral_deg_imp)) / 4))
