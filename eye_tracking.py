# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:10:33 2019

@author: s144451
"""

# DONT CHANGE BELOW
################################
# Preface here
#
# from psychopy import prefs, visual, core, event, monitors, tools, logging
import tobii_research as tr
import sys


#
#
#
################################
# SETUP HERE
#

class EyeTracking:

    global_gaze_data = []
    halted = False
    
    licence_file = "licenses/license_key_00395217_-_DTU_Compute_IS404-100106342114"
    #gazedata_filename = "gaze_data/"+datetime.datetime.now().strftime("%A, %d. %B %Y %I.%M.%S %p")+".csv"
    
    current_target = (0.5, 0.5)
    

    channels = 31 # count of the below channels, incl. those that are 3 or 2 long
    gaze_params = [
        'device_time_stamp',
        'left_gaze_origin_in_trackbox_coordinate_system',
        'left_gaze_origin_in_user_coordinate_system',
        'left_gaze_origin_validity',
        'left_gaze_point_in_user_coordinate_system',
        'left_gaze_point_on_display_area',
        'left_gaze_point_validity',
        'left_pupil_diameter',
        'left_pupil_validity',
        'right_gaze_origin_in_trackbox_coordinate_system',
        'right_gaze_origin_in_user_coordinate_system',
        'right_gaze_origin_validity',
        'right_gaze_point_in_user_coordinate_system',
        'right_gaze_point_on_display_area',
        'right_gaze_point_validity',
        'right_pupil_diameter',
        'right_pupil_validity',
        'system_time_stamp',
        'current_target_point_on_display_area'
    ]

    def __init__(self):
        # Find Eye Tracker and Apply License (edit to suit actual tracker serial no)
        ft = tr.find_all_eyetrackers()
        if len(ft) == 0:
            raise Exception("No eye trackers found")
        
        for tracker in ft:
            print("Found Tobii Tracker at '%s'" % (tracker.address))
        
        # Pick first tracker
        mt = ft[0]
        print("Using Tobii Tracker at '%s'" % (mt.address))
        
        # Apply license
        if self.license_file != "":
            with open(self.license_file, "rb") as f:
                license = f.read()
        
                res = mt.apply_licenses(license)
                if len(res) == 0:
                    print("Successfully applied license from single key")
                else:
                    print("Failed to apply license from single key. Validation result: %s." % (res[0].validation_result))
                    sys.exit
        else:
            print("No license file installed")
        
        self.mt = mt

    def gaze_data_callback(self, gaze_data):
        '''send gaze data'''
    
        '''
        This is what we get from the tracker:
        device_time_stamp
        left_gaze_origin_in_trackbox_coordinate_system (3)
        left_gaze_origin_in_user_coordinate_system (3)
        left_gaze_origin_validity
        left_gaze_point_in_user_coordinate_system (3)
        left_gaze_point_on_display_area (2)
        left_gaze_point_validity
        left_pupil_diameter
        left_pupil_validity
        right_gaze_origin_in_trackbox_coordinate_system (3)
        right_gaze_origin_in_user_coordinate_system (3)
        right_gaze_origin_validity
        right_gaze_point_in_user_coordinate_system (3)
        right_gaze_point_on_display_area (2)
        right_gaze_point_validity
        right_pupil_diameter
        right_pupil_validity
        system_time_stamp
        '''
        
        try:
            
            gaze_data.append(self.current_target)
            self.global_gaze_data.append(gaze_data)
            
            # print(unpack_gaze_data(gaze_data)
        except:
            print("Error in callback: ")
            print(sys.exc_info())
            
            global halted
            halted = True
    
    
    def start_gaze_tracking(self): 
        self.mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
    
    def end_gaze_tracking(self):
        self.mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        

    def set_current_target(self, x, y):
        self.current_target = (x, y)
        print("Target x: " + x)
        print("Target y: " + y)
        





