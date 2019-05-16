#
# Tobii controller for PsychoPy
# 
# author: Hiroyuki Sogo
# Distributed under the terms of the GNU General Public License v3 (GPLv3).
# 

from __future__ import division
from __future__ import absolute_import

import types
import datetime
import numpy as np
import time
import warnings
import math

import tobii_research

try:
    import Image
    import ImageDraw
except:
    from PIL import Image
    from PIL import ImageDraw

import psychopy.visual
import psychopy.event
import psychopy.core
import psychopy.monitors
import psychopy.logging
import psychopy.sound

class tobii_controller:

    """
    Default estimates
    """
    dist_to_screen = 60
    screen_width = 1200
    screen_height = 800

    """
    PsychoPy specfications
    """
    psychopy.logging.console.setLevel(psychopy.logging.CRITICAL)    # IGNORE UNSAVED MONITOR WARNINGS IN CONSOLE    
    default_background_color = [-1,-1,-1]
    is_mouse_enabled = False
    
    rot_deg_per_frame = 3     # how many degrees of rotation per frame
    
    
    default_calibration_target_dot_size = {
            'pix': 2.0, 'norm':0.004, 'height':0.002, 'cm':0.05,
            'deg':0.05, 'degFlat':0.05, 'degFlatPos':0.05
        }
    default_calibration_target_disc_size = {
            'pix': 2.0*20, 'norm':0.004*20, 'height':0.002*20, 'cm':0.05*20,
            'deg':0.05*20, 'degFlat':0.05*20, 'degFlatPos':0.05*20
        }
    
    default_key_index_dict = {
            '1':0, 'num_1':0, '2':1, 'num_2':1, '3':2, 'num_3':2,
            '4':3, 'num_4':3, '5':4, 'num_5':4, '6':5, 'num_6':5,
            '7':6, 'num_7':6, '8':7, 'num_8':7, '9':8, 'num_9':8
        }

    
    """
    Tobii controller for PsychoPy
    tobii_research package is required to use this class.
    """
    eyetracker = None
    calibration = None
    win = None
    control_window = None
    gaze_data = []
    event_data = []
    retry_points = []
    datafile = None
    embed_events = False
    recording = False
    key_index_dict = default_key_index_dict.copy()

    # Tobii data collection parameters
    subscribe_to_data = False
    do_reset_recording = True
    current_target = (0.5, 0.5)
    global_gaze_data = []    
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
    
#    license_file = "licenses/license_key_00395217_-_DTU_Compute_IS404-100106342114" #lab
    license_file = "licenses/license_key_00395217_-_DTU_Compute_IS404-100106241134" #home
    
    def __init__(self, screen_width, screen_height, eyetracker_id=0):
        """
        Initialize tobii_controller object.
        
        :param win: PsychoPy Window object.
        :param int id: ID of Tobii unit to connect with.
            Default value is 0.
        """
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.sound = psychopy.sound.Sound('sounds/baby_einstein.wav')
            
        self.set_up_eyetracker(eyetracker_id)
        
    def set_up_eyetracker(self, eyetracker_id=0):
        eyetrackers = tobii_research.find_all_eyetrackers()
        

        if len(eyetrackers)==0:
            print('No Tobii eyetrackers')
        
        else:
            try:
                self.eyetracker = eyetrackers[eyetracker_id]
                
                with open(self.license_file, "rb") as f:
                    license = f.read()
                    res = self.eyetracker.apply_licenses(license)
                    if len(res) == 0:
                        print("Successfully applied license from single key")
                    else:
                        print("Failed to apply license from single key. Validation result: %s." % (res[0].validation_result))
                        
            except:
                raise ValueError('Invalid eyetracker ID {}\n({} eyetrackers found)'.format(eyetracker_id, len(eyetrackers)))
        
            if self.is_eye_tracker_on():
                self.calibration = tobii_research.ScreenBasedCalibration(self.eyetracker)
            else:
                self.eyetracker = None
        
    def is_eye_tracker_on(self):
        self.subscribe_dict()
        self.start_recording()
        time.sleep(1)
        self.stop_recording()
        self.unsubscribe_dict()         
        return len(self.global_gaze_data) > 0
        
    def set_dist_to_screen(self, dist_to_screen):
        self.dist_to_screen = dist_to_screen

    def play_sound(self):
        self.sound.play()
        
    def pause_sound(self):
        self.sound.stop()
        
    def cm2deg(self, cm, monitor, correctFlat=False):
        """
        Bug-fixed version of psychopy.tools.monitorunittools.cm2deg
        (PsychoPy version<=1.85.1).
        """
        
        if not isinstance(monitor, psychopy.monitors.Monitor):
            msg = ("cm2deg requires a monitors.Monitor object as the second "
                   "argument but received %s")
            raise ValueError(msg % str(type(monitor)))
        dist = monitor.getDistance()
        if dist is None:
            msg = "Monitor %s has no known distance (SEE MONITOR CENTER)"
            raise ValueError(msg % monitor.name)
        if correctFlat:
            return np.degrees(np.arctan(cm / dist))
        else:
            return cm / (dist * 0.017455)
    
    
    def pix2deg(self, pixels, monitor, correctFlat=False):
        """
        Bug-fixed version of psychopy.tools.monitorunittools.pix2deg
        (PsychoPy version<=1.85.1).
        """
        
        scrWidthCm = monitor.getWidth()
        scrSizePix = monitor.getSizePix()
        if scrSizePix is None:
            msg = "Monitor %s has no known size in pixels (SEE MONITOR CENTER)"
            raise ValueError(msg % monitor.name)
        if scrWidthCm is None:
            msg = "Monitor %s has no known width in cm (SEE MONITOR CENTER)"
            raise ValueError(msg % monitor.name)
        cmSize = pixels * float(scrWidthCm) / scrSizePix[0]
        return self.cm2deg(cmSize, monitor, correctFlat)


    def make_psycho_window(self, background_color=None, screen=1):
        self.bg_color = background_color
        
        # make a new monitor for the window - ignore the warning (we dont store any calibrations for this monitor)
        mon = psychopy.monitors.Monitor('MyScreen')
        
        width = self.screen_width if screen == 1 else 700
        height = self.screen_width if screen == 1 else 500
            
        mon.setDistance(self.dist_to_screen)
        mon.setSizePix((width, height))
        
        bg = self.bg_color if self.bg_color != None else self.default_background_color
        
        if screen == 1: 
            self.win = psychopy.visual.Window(size=(self.screen_width, self.screen_height), screen=screen, fullscr=True, units='norm', monitor=mon)
            self.win.setColor(bg, colorSpace='rgb')
            psychopy.event.Mouse(visible=self.is_mouse_enabled, win=self.win)
        if screen == 0:
            self.control_window = psychopy.visual.Window(size=(width, height), screen=screen, fullscr=False, units='norm', monitor=mon, pos = [1920-width-10,1080/4])
            self.control_window.setColor(bg, colorSpace='rgb')
            print(self.control_window.pos)
        
        
        
    def close_psycho_window(self, screen=1):
        self.bg_color = None    # reset color scheme
        if screen == 1:
            self.win.winHandle.set_fullscreen(False) # disable fullscreen
            self.win.close()
        elif screen == 0:
#            self.control_window.winHandle.set_fullscreen(False) # disable fullscreen
            self.control_window.close()
            
     
        
    def show_status_admin(self, text_color='white', enable_mouse=False, screen=1):
        """
        Draw eyetracker status on the screen.
        
        :param text_color: Color of message text. Default value is 'white'
        :param bool enable_mouse: If True, mouse operation is enabled.
            Default value is False.
        """
        
        self.make_psycho_window(background_color="gray", screen=screen)
        
        window = self.win if screen == 1 else self.control_window
        
#        if enable_mouse == False:
#            mouse = psychopy.event.Mouse(visible=False, win=self.win)
        
        self.gaze_data_status = None
        
        msg = psychopy.visual.TextStim(window, color=text_color,
            height=0.02, pos=(0,-0.35), units='height', autoLog=False, text="No eye tracker data detected")
        bgrect = psychopy.visual.Rect(window,
            width=0.6, height=0.6, lineColor='white', fillColor='black',
            units='height', autoLog=False)
        leye = psychopy.visual.Circle(window,
            size=0.05, units='height', lineColor=None, fillColor='green',
            autoLog=False)
        reye = psychopy.visual.Circle(window, size=0.05, units='height',
            lineColor=None, fillColor='red', autoLog=False)
        
        b_show_status = True
        while b_show_status:
            bgrect.draw()
            if self.gaze_data_status is not None:
                lp, lv, rp, rv = self.gaze_data_status
                msgst = 'Left: {:.3f},{:.3f},{:.3f}\n'.format(*lp)
                msgst += 'Right: {:.3f},{:.3f},{:.3f}\n'.format(*rp)
                msg.setText(msgst)
                if lv:
                    leye.setPos(((1-lp[0]-0.5)/2,(1-lp[1]-0.5)/2))
                    leye.setRadius((1-lp[2])/2)
                    leye.draw()
                if rv:
                    reye.setPos(((1-rp[0]-0.5)/2,(1-rp[1]-0.5)/2))
                    reye.setRadius((1-rp[2])/2)
                    reye.draw()
            
            for key in psychopy.event.getKeys():
                if key == 'escape' or key == 'space':
                    b_show_status = False
            
#            if enable_mouse and mouse.getPressed()[0]:
#                b_show_status = False
            
            msg.draw()
            window.flip()
            
        
        self.close_psycho_window(screen=screen)
        
        
        
    def show_status(self, text_color='white', enable_mouse=False, screen=1):
        """
        Draw eyetracker status on the screen.
        
        :param text_color: Color of message text. Default value is 'white'
        :param bool enable_mouse: If True, mouse operation is enabled.
            Default value is False.
        """
        
        self.make_psycho_window(background_color="gray", screen=screen)
        
        window = self.win if screen == 1 else self.control_window
        
#        if enable_mouse == False:
#            mouse = psychopy.event.Mouse(visible=False, win=self.win)
        
        self.gaze_data_status = None
#        if self.eyetracker is not None:
#            self.eyetracker.subscribe_to(tobii_research.EYETRACKER_GAZE_DATA, self.on_gaze_data_status)
        
        msg = psychopy.visual.TextStim(window, color=text_color,
            height=0.02, pos=(0,-0.35), units='height', autoLog=False, text="No eye tracker data detected")
        bgrect = psychopy.visual.Rect(window,
            width=0.6, height=0.6, lineColor='white', fillColor='black',
            units='height', autoLog=False)
        leye = psychopy.visual.Circle(window,
            size=0.05, units='height', lineColor=None, fillColor='green',
            autoLog=False)
        reye = psychopy.visual.Circle(window, size=0.05, units='height',
            lineColor=None, fillColor='red', autoLog=False)
        
        b_show_status = True
        while b_show_status:
            bgrect.draw()
            if self.gaze_data_status is not None:
                lp, lv, rp, rv = self.gaze_data_status
                msgst = 'Left: {:.3f},{:.3f},{:.3f}\n'.format(*lp)
                msgst += 'Right: {:.3f},{:.3f},{:.3f}\n'.format(*rp)
                msg.setText(msgst)
                if lv:
                    leye.setPos(((1-lp[0]-0.5)/2,(1-lp[1]-0.5)/2))
                    leye.setRadius((1-lp[2])/2)
                    leye.draw()
                if rv:
                    reye.setPos(((1-rp[0]-0.5)/2,(1-rp[1]-0.5)/2))
                    reye.setRadius((1-rp[2])/2)
                    reye.draw()
            
            for key in psychopy.event.getKeys():
                if key == 'escape' or key == 'space':
                    b_show_status = False
            
#            if enable_mouse and mouse.getPressed()[0]:
#                b_show_status = False
            
#            msg.draw()
            window.flip()
            
            
#        if self.eyetracker is not None:
#            self.eyetracker.unsubscribe_from(tobii_research.EYETRACKER_GAZE_DATA)
        
        self.close_psycho_window(screen=screen)
        
    def on_gaze_data_status(self, gaze_data):
        """
        Callback function used by
        :func:`~psychopy_tobii_controller.tobii_controller.show_status`
        
        Usually, users don't have to call this method.
        """
        
        lp = gaze_data.left_eye.gaze_origin.position_in_track_box_coordinates
        lv = gaze_data.left_eye.gaze_origin.validity
        rp = gaze_data.right_eye.gaze_origin.position_in_track_box_coordinates
        rv = gaze_data.right_eye.gaze_origin.validity
        self.gaze_data_status = (lp, lv, rp, rv)

    def start_custom_calibration(self, num_points=2, stim_type="default", stimuli_path="stimuli/smiley_yellow.png"):
            
        # Run calibration.
        target_points = [(-0.5, 0.0), (0.5, 0.0)]
        if num_points == 5:
            target_points = [(-0.4,0.4), (0.4,0.4), (0.0,0.0), (-0.4,-0.4), (0.4,-0.4)]
        self.run_calibration(target_points, stim_type=stim_type, stimuli_path="stimuli/smiley_yellow.png")       
        
        
        # THIS CODE MAKES A GAZE TRACE AFTER THE CALIBRATION
#        # If calibration is aborted by pressing ESC key, return value of run_calibration()
#        # is 'abort'.
#        if ret != 'abort':
#            
#            marker = psychopy.visual.Rect(self.win, width=0.01, height=0.01)
#            
#            # Start recording.
#            self.subscribe()
#            waitkey = True
#            while waitkey:
#                # Get the latest gaze position data.
#                currentGazePosition = self.get_current_gaze_position()
#                
#                # Gaze position is a tuple of four values (lx, ly, rx, ry).
#                # The value is numpy.nan if Tobii failed to detect gaze position.
#                if not np.nan in currentGazePosition:
#                    marker.setPos(currentGazePosition[0:2])
#                    marker.setLineColor('white')
#                else:
#                    marker.setLineColor('red')
#                keys = psychopy.event.getKeys ()
#                if 'space' in keys:
#                    waitkey=False
#                elif len(keys)>=1:
#                    # Record the first key name to the data file.
#                    self.record_event(keys[0])
#                
#                marker.draw()
#                self.win.flip()
#            # Stop recording.
#            self.unsubscribe()
#            # Close the data file.
#            self.close_datafile()
        
#        self.close_psycho_window()
        
        
    def run_calibration(self, calibration_points, move_duration=1.5,
            shuffle=True, start_key='space', decision_key='space',
            text_color='white', enable_mouse=False, stim_type="default", stimuli_path="stimuli/smiley_yellow.png"):
        """
        Run calibration.
        
        :param calibration_points: List of position of calibration points.
        :param float move_duration: Duration of animation of calibration target.
            Unit is second.  Default value is 1.5.
        :param bool shuffle: If True, order of calibration points is shuffled.
            Otherwise, calibration target moves in the order of calibration_points.
            Default value is True.
        :param str start_key: Name of key to start calibration procedure.
            If None, calibration starts immediately afte this method is called.
            Default value is 'space'.
        :param str decision_key: Name of key to accept/retry calibration.
            Default value is 'space'.
        :param text_color: Color of message text. Default value is 'white'
        :param bool enable_mouse: If True, mouse operation is enabled.
            Default value is False.
        """
        
        # set sizes and init calibration
        self.calibration_target_dot_size = self.default_calibration_target_dot_size[self.win.units]
        self.calibration_target_disc_size = self.default_calibration_target_disc_size[self.win.units]
        self.calibration_target_dot = psychopy.visual.Circle(self.win, 
            radius=self.calibration_target_dot_size, fillColor='white',
            lineColor=None,lineWidth=1, autoLog=False)
        self.calibration_target_disc = psychopy.visual.Circle(self.win,
            radius=self.calibration_target_disc_size, fillColor='lime',
            lineColor='white', lineWidth=1, autoLog=False)
        self.update_calibration = self.update_calibration_default
        if self.win.units == 'norm': # fix oval
            self.calibration_target_dot.setSize([float(self.win.size[1])/self.win.size[0], 1.0])
            self.calibration_target_disc.setSize([float(self.win.size[1])/self.win.size[0], 1.0])
        
        
        if not (2 <= len(calibration_points) <= 9):
            raise ValueError('Calibration points must be 2~9')
        
        
        if enable_mouse == False:
            mouse = psychopy.event.Mouse(visible=False, win=self.win)
        
        img = Image.new('RGBA',tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        
        result_img = psychopy.visual.SimpleImageStim(self.win, img, autoLog=False)
        result_msg = psychopy.visual.TextStim(self.win, pos=(0,-self.win.size[1]/4),
            color=text_color, units='pix', autoLog=False)
        
       
        remove_marker = psychopy.visual.Circle(
            self.win, radius=self.calibration_target_dot.radius*5,
            fillColor='black', lineColor='white', lineWidth=1, autoLog=False)
        if self.win.units == 'norm': # fix oval
            remove_marker.setSize([float(self.win.size[1])/self.win.size[0], 1.0])
            remove_marker.setSize([float(self.win.size[1])/self.win.size[0], 1.0])
             
        if self.eyetracker is not None:
            self.calibration.enter_calibration_mode()

        self.move_duration = move_duration
        self.original_calibration_points = calibration_points[:]
        self.retry_points = list(range(len(self.original_calibration_points))) # set all points
        
        in_calibration_loop = True
        while in_calibration_loop:
            self.calibration_points = []
            for i in range(len(self.original_calibration_points)):
                if i in self.retry_points:
                    self.calibration_points.append(self.original_calibration_points[i])
            
            if shuffle:
                np.random.shuffle(self.calibration_points)
            
            if start_key is not None or enable_mouse:
                waitkey = False
                if start_key is not None:
                    if enable_mouse == True:
                        result_msg.setText('Press {} or click left button to start calibration'.format(start_key))
                    else:
                        result_msg.setText('Press {} to start calibration'.format(start_key))
                else: # enable_mouse==True
                    result_msg.setText('Click left button to start calibration')
                while waitkey:
                    for key in psychopy.event.getKeys():
                        if key==start_key:
                           waitkey = False
                    
                    if enable_mouse and mouse.getPressed()[0]:
                        waitkey = False
                    
                    result_msg.draw()
                    self.win.flip()
            else:
                self.win.flip()
            
            if stim_type == "default":
                self.update_calibration()
            elif stim_type == "img":
                self.update_calibration_img(stimuli_path)
                
            calibration_result = None
            if self.eyetracker is not None:
                calibration_result = self.calibration.compute_and_apply()
            
            self.win.flip()
            
            img_draw.rectangle(((0,0),tuple(self.win.size)),fill=(0,0,0,0))
            if calibration_result is None or calibration_result.status == tobii_research.CALIBRATION_STATUS_FAILURE:
                #computeCalibration failed.
                pass
            
            else:
                if len(calibration_result.calibration_points) == 0:
                    pass
                else:
                    for calibration_point in calibration_result.calibration_points:
                        p = calibration_point.position_on_display_area
                        for calibration_sample in calibration_point.calibration_samples:
                            lp = calibration_sample.left_eye.position_on_display_area
                            rp = calibration_sample.right_eye.position_on_display_area
                            if calibration_sample.left_eye.validity == tobii_research.VALIDITY_VALID_AND_USED:
                                img_draw.line(((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                                              (lp[0]*self.win.size[0], lp[1]*self.win.size[1])), fill=(0,255,0,255))
                            if calibration_sample.right_eye.validity == tobii_research.VALIDITY_VALID_AND_USED:
                                img_draw.line(((p[0]*self.win.size[0], p[1]*self.win.size[1]),
                                              (rp[0]*self.win.size[0], rp[1]*self.win.size[1])), fill=(255,0,0,255))
                        img_draw.ellipse(((p[0]*self.win.size[0]-3, p[1]*self.win.size[1]-3),
                                         (p[0]*self.win.size[0]+3, p[1]*self.win.size[1]+3)), outline=(0,0,0,255))

            if enable_mouse == False:
                result_msg.setText('Accept/Retry: {} or right-click\nSelect recalibration points: 0-9 key or left-click'.format(decision_key))
            else:
                result_msg.setText('Accept/Retry: {}\nSelect recalibration points: 0-9 key'.format(decision_key))
            result_img.setImage(img)
            
            waitkey = True
            self.retry_points = []
            if enable_mouse == False:
                mouse.setVisible(True)
            while waitkey:
                for key in psychopy.event.getKeys():
                    if key in [decision_key, 'escape']:
                        waitkey = False
                    elif key in ['0', 'num_0']:
                        if len(self.retry_points) == 0:
                            self.retry_points = list(range(len(self.original_calibration_points)))
                        else:
                            self.retry_points = []
                    elif key in self.key_index_dict:
                        key_index = self.key_index_dict[key]
                        if key_index<len(self.original_calibration_points):
                            if key_index in self.retry_points:
                                self.retry_points.remove(key_index)
                            else:
                                self.retry_points.append(key_index)
                if enable_mouse == False:
                    pressed = mouse.getPressed()
                    if pressed[2]: # right click
                        key = decision_key
                        waitkey = False
                    elif pressed[0]: # left click
                        mouse_pos = mouse.getPos()
                        for key_index in range(len(self.original_calibration_points)):
                            p = self.original_calibration_points[key_index]
                            if np.linalg.norm([mouse_pos[0]-p[0], mouse_pos[1]-p[1]]) < self.calibration_target_dot.radius*5:
                                if key_index in self.retry_points:
                                    self.retry_points.remove(key_index)
                                else:
                                    self.retry_points.append(key_index)
                                time.sleep(0.2)
                                break
                result_img.draw()
                if len(self.retry_points)>0:
                    for index in self.retry_points:
                        if index > len(self.original_calibration_points):
                            self.retry_points.remove(index)
                        remove_marker.setPos(self.original_calibration_points[index])
                        remove_marker.draw()
                result_msg.draw()
                self.win.flip()
        
            if key == decision_key:
                if len(self.retry_points) == 0:
#                    retval = 'accept'
                    in_calibration_loop = False
                else: #retry
                    for point_index in self.retry_points:
                        x, y = self.get_tobii_pos(self.original_calibration_points[point_index])
                        if self.eyetracker is not None:
                            self.calibration.discard_data(x, y)
            elif key == 'escape':
#                retval = 'abort'
                in_calibration_loop = False
            else:
                raise RuntimeError('Calibration: Invalid key')
                
            if enable_mouse == False:
                mouse.setVisible(False)
        
        if self.eyetracker is not None:
            self.calibration.leave_calibration_mode()

        if enable_mouse == False:
            mouse.setVisible(False)

    def flash_screen(self):     
        
        r = self.win.color[0]
        g = self.win.color[1]
        b = self.win.color[2]
        
        while r <= 1:
            r += 0.05
            g += 0.05
            b += 0.05
            self.win.setColor((r,g,b), colorSpace='rgb')
            psychopy.core.wait(0.05)
            self.win.flip()
        
        while r >= -1:
            r -= 0.05
            g -= 0.05
            b -= 0.05
            self.win.setColor((r,g,b), colorSpace='rgb')
            psychopy.core.wait(0.05)
            self.win.flip()
        
        
    def animate_test(self, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points, stimuli_paths=["stimuli/smiley_yellow.png"], frame_delay=0.015):
        
        self.make_psycho_window()
        
        img_stims = []
        for stimuli_path in stimuli_paths:
            img = Image.open(stimuli_path)
            img_stim = psychopy.visual.ImageStim(self.win, image=img, autoLog=False)
            img_stim.size = (0.15, 0.15)
            img_stims.append(img_stim)        
        
        for i, (gaze_point_left, gaze_point_right, gaze_point_left_corrected, gaze_point_right_corrected, target_point) in enumerate(zip(gaze_data_left.T, gaze_data_right.T, gaze_data_left_corrected.T, gaze_data_right_corrected.T, target_points.T)):
            
            target_point = self.get_psychopy_pos(target_point)
            gaze_point_left = self.get_psychopy_pos(gaze_point_left)
            gaze_point_right = self.get_psychopy_pos(gaze_point_right)
            gaze_point_left_corrected = self.get_psychopy_pos(gaze_point_left_corrected)
            gaze_point_right_corrected = self.get_psychopy_pos(gaze_point_right_corrected)
            
            img_stim = img_stims[(i - 1) % len(img_stims)]
            img_stim.setPos(target_point)
            img_stim.ori = i * self.rot_deg_per_frame
            img_stim.draw()

            stim_left = psychopy.visual.Circle(self.win, radius=0.05, fillColor='red', autoLog=False)
            stim_left.setPos(gaze_point_left)
            stim_left.draw()

            stim_right = psychopy.visual.Circle(self.win, radius=0.05, fillColor='green', autoLog=False)
            stim_right.setPos(gaze_point_right)
            stim_right.draw()

            stim_left_corrected= psychopy.visual.Circle(self.win, radius=0.05, fillColor='blue', autoLog=False)
            stim_left_corrected.setPos(gaze_point_left_corrected)
            stim_left_corrected.draw()
            
            stim_right_corrected = psychopy.visual.Circle(self.win, radius=0.05, fillColor='purple', autoLog=False)
            stim_right_corrected.setPos(gaze_point_right_corrected)
            stim_right_corrected.draw()

            self.win.flip()
            
            psychopy.core.wait(frame_delay)
        
        self.close_psycho_window(screen=1)
        
    
#    def make_transformation(self, stimuli_path="stimuli/smiley_yellow.png", enable_mouse=False):        
#        
#        img = Image.open(stimuli_path)
#        img_stim = psychopy.visual.ImageStim(self.win, image=img, autoLog=False)
#        img_stim.size = (0.15,0.15)
#                
#        img_positions = [(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)]
#        np.random.shuffle(img_positions)
#
#        self.subscribe_dict()
#        clock = psychopy.core.Clock()
#        
#        for img_pos in img_positions:       
#            self.current_target = self.get_tobii_pos(img_pos)
#            
#            i = 0
#            clock.reset()
#            current_time = clock.getTime()
#            while current_time < 3:
#                img_stim.setPos(img_pos)
#                img_stim.ori = i * self.rot_deg_per_frame
#                img_stim.draw()
#                self.win.flip()
#                
#                i += 1
#                psychopy.core.wait(0.015)
#                current_time = clock.getTime()
#        
#        self.unsubscribe_dict()
        

    def start_fixation_exercise(self, positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"], frame_delay=0.015, fixation_duration = 3):
        
        img_stims = []
        for stimuli_path in stimuli_paths:
            img = Image.open(stimuli_path)
            img_stim = psychopy.visual.ImageStim(self.win, image=img, autoLog=False)
            img_stim.size = (0.15, 0.15)
            img_stims.append(img_stim)        
        
        np.random.shuffle(positions)
        
#        self.subscribe_dict()
        self.start_recording()
        clock = psychopy.core.Clock()
        
        pos_index = 0
        for pos in positions:
            self.current_target = self.get_tobii_pos(pos)
            i = 0
            clock.reset()
            current_time = clock.getTime()
            while current_time < fixation_duration:
                img_stim = img_stims[(pos_index - 1) % len(img_stims)]
                img_stim.setPos(pos)
                img_stim.ori = i * self.rot_deg_per_frame
                img_stim.draw()
                self.win.flip()
                
                i += 1
                
                psychopy.core.wait(frame_delay)
                current_time = clock.getTime()
                
            pos_index += 1
        
#        self.unsubscribe_dict()
        self.stop_recording()
    
    
    
    
    def start_fixation_exercise_animate_transition(self, positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"], frame_delay=0.015, move_duration=1, fixation_duration = 3):
        
        img_stims = []
        for stimuli_path in stimuli_paths:
            img = Image.open(stimuli_path)
            img_stim = psychopy.visual.ImageStim(self.win, image=img, autoLog=False)
            img_stim.size = (0.15, 0.15)
            img_stims.append(img_stim)        
        
        np.random.shuffle(positions)
        
        position_pairs = [[positions[i], positions[i+1]] for i in range(len(positions)-1)]
        
        
        
#        self.subscribe_dict()
        self.start_recording()
        clock = psychopy.core.Clock()
        
        pos_index = 0
        for pos in positions:
            self.current_target = self.get_tobii_pos(pos)
            i = 0
            clock.reset()
            current_time = clock.getTime()
            while current_time < fixation_duration:
                img_stim = img_stims[(pos_index - 1) % len(img_stims)]
                img_stim.setPos(pos)
                img_stim.ori = i * self.rot_deg_per_frame
                img_stim.draw()
                self.win.flip()
                
                i += 1
                
                psychopy.core.wait(frame_delay)
                current_time = clock.getTime()
                
            if pos_index < len(position_pairs):
#                self.subscribe_to_data = False
                self.do_reset_recording = False
                self.start_pursuit_exercise(pathing="linear", positions=position_pairs[pos_index], stimuli_paths=stimuli_paths, frame_delay=frame_delay, move_duration=move_duration)
#                self.subscribe_to_data = True
                self.do_reset_recording = True
            pos_index += 1
            
            
#        self.unsubscribe_dict()
        self.stop_recording()
        
    
    def calc_pursuit_route(self, pathing, positions, frame_delay=0.03, move_duration=5, reverse=False):
        
        
        # Normal coordinate system
        intermediate_positions = []
        move_steps = move_duration / frame_delay
        
        if pathing == "linear":
            
            total_dist = 0
            for i in range(len(positions) - 1):
                total_dist += self.get_euclidean_distance(positions[i], positions[i + 1])
            
            # intermediate points
            for i in range(len(positions)):
                if i+1 < len(positions):
                    start_pos = positions[i]
                    end_pos = positions[i+1]
                                
                    euc_dist = self.get_euclidean_distance(start_pos, end_pos)
                    amount_of_path = euc_dist / total_dist
                    move_steps_for_path = amount_of_path * move_steps
                    
                    intermediate_positions.extend(self.get_equidistant_points(start_pos, end_pos, move_steps_for_path))
                    
        elif pathing == "circle" and len(positions) == 2:
            start_pos = positions[0]
            center_pos = positions[1]
            
            intermediate_positions.append(start_pos)
            
            r = ((start_pos[0] - center_pos[0]) ** 2 + (start_pos[1] - center_pos[1]) ** 2) ** 0.5
            
            theta_x = math.acos(start_pos[0] / r)
            theta_y = math.asin(start_pos[1] / r)
            
            theta = theta_x if theta_y >= 0 else -theta_x
            
            delta_theta = 2*math.pi / move_steps
            
            step = 0
            while move_steps > step:
                step = step + 1
                theta = theta + delta_theta
                pos = (r*math.cos(theta), r*math.sin(theta))
                intermediate_positions.append(pos)
                
        elif pathing == "spiral" and len(positions) == 2:
            start_pos = positions[0]
            end_pos = positions[1]
            
            intermediate_positions.append(start_pos)
            
            r = ((start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2) ** 0.5
            
            theta_x = math.acos(start_pos[0] / r)
            theta_y = math.asin(start_pos[1] / r)
            
            theta = theta_x if theta_y >= 0 else -theta_x
            
            dr = r / move_steps
            
            while r >= 0:
                r -= dr
                theta = theta + (0.05 * math.pi) / (r * (move_duration + 1/r))
                pos = (r*math.cos(theta), r*math.sin(theta))
                intermediate_positions.append(pos)
        
        if reverse:
            intermediate_positions.reverse()
        return intermediate_positions
        
    
    def start_pursuit_exercise(self, pathing="linear", positions=[(-0.7,0.0),(0.0,0.0)], stimuli_paths=["stimuli/smiley_yellow.png"], reverse=False, frame_delay=0.011, move_duration=5):
        
        img_stims = []
        for stimuli_path in stimuli_paths:
            img = Image.open(stimuli_path)
            img_stim = psychopy.visual.ImageStim(self.win, image=img, autoLog=False)
            img_stim.size = (0.15, 0.15)
            img_stims.append(img_stim)
        
#        frame_delay = 0.015
        intermediate_positions = self.calc_pursuit_route(pathing, positions=positions, frame_delay=frame_delay, move_duration=move_duration, reverse=reverse)
        
        if self.do_reset_recording:
#            self.subscribe_dict()
            self.start_recording()
        
        pos_index = 0
        for i, pos in enumerate(intermediate_positions):
            img_stim = img_stims[(pos_index) % len(img_stims)]
            img_stim.setPos(pos)
            img_stim.ori = i * self.rot_deg_per_frame
            img_stim.opacity = 1.0
            img_stim.draw()
            
            if pathing == "spiral":
                img_stim = img_stims[(pos_index + 1) % len(img_stims)]
                img_stim.setPos(pos)
                img_stim.ori = i * self.rot_deg_per_frame
                img_stim.opacity = (i % int(len(intermediate_positions) / len(img_stims) + 1)) / int(len(intermediate_positions) / len(img_stims))
                img_stim.draw()
                
            self.current_target = self.get_tobii_pos(pos)
            self.win.flip()
            
            if pathing == "linear" and pos[0] == positions[pos_index + 1][0] and pos[1] == positions[pos_index + 1][1]:
                pos_index += 1
            
            if pathing == "spiral" and i % int(len(intermediate_positions) / len(img_stims)) == 0 and i > 0:
                pos_index += 1
            
            psychopy.core.wait(frame_delay)
            
        if self.do_reset_recording:
#            self.unsubscribe_dict()
            self.stop_recording()

    def get_euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0])**2+(p1[1] - p2[1])**2)**0.5

    def get_equidistant_points(self, p1, p2, parts):
        return zip(np.linspace(p1[0], p2[0], parts), np.linspace(p1[1], p2[1], parts))
    
    def collect_calibration_data(self, p, cood='PsychoPy'):
        """
        Callback function used by
        :func:`~psychopy_tobii_controller.tobii_controller.run_calibration`
        
        Usually, users don't have to call this method.
        """
        
        if cood=='PsychoPy':
            self.calibration.collect_data(*self.get_tobii_pos(p))
        elif cood =='Tobii':
            self.calibration.collect_data(*p)
        else:
            raise ValueError('cood must be \'PsychoPy\' or \'Tobii\'')


    def update_calibration_default(self):
        """
        Updating calibration target and correcting calibration data.
        This method is called by
        :func:`~psychopy_tobii_controller.tobii_controller.run_calibration`
        
        Usually, users don't have to call this method.
        """
        
        clock = psychopy.core.Clock()
        for point_index in range(len(self.calibration_points)):
            x, y = self.get_tobii_pos(self.calibration_points[point_index])
            self.calibration_target_dot.setPos(self.calibration_points[point_index])
            self.calibration_target_disc.setPos(self.calibration_points[point_index])
            
            clock.reset()
            current_time = clock.getTime()
            while current_time < self.move_duration:
                self.calibration_target_disc.setRadius(
                    (self.calibration_target_dot_size*2.0-self.calibration_target_disc_size)/ \
                     self.move_duration*current_time+self.calibration_target_disc_size
                    )
                psychopy.event.getKeys()
                self.calibration_target_disc.draw()
                self.calibration_target_dot.draw()
                self.win.flip()
                current_time = clock.getTime()
            
            if self.eyetracker is not None:
                self.calibration.collect_data(x, y)
                
    def update_calibration_img(self, stimuli_path):

        stim_img = Image.open(stimuli_path)        
        stimuli = psychopy.visual.ImageStim(self.win, image=stim_img, autoLog=False)
        stimuli.size = (0.15,0.15)        
        
        position_pairs = [[self.calibration_points[i], self.calibration_points[i+1]] for i in range(len(self.calibration_points)-1)]
  
                
        clock = psychopy.core.Clock()
        for point_index in range(len(self.calibration_points)):
            x, y = self.get_tobii_pos(self.calibration_points[point_index])
            i = 0
            
            clock.reset()
            current_time = clock.getTime()
            while current_time < self.move_duration:
                psychopy.event.getKeys()
                stimuli.setPos(self.calibration_points[point_index])
                stimuli.ori = i * self.rot_deg_per_frame
                stimuli.draw()
                self.win.flip()
                
                i += 1
                psychopy.core.wait(0.015)
                current_time = clock.getTime()
            
            if self.eyetracker is not None:
                self.calibration.collect_data(x, y)
            
            if point_index < len(position_pairs):
                self.do_reset_recording = False
                self.start_pursuit_exercise(pathing="linear", positions=position_pairs[point_index], stimuli_paths=[stimuli_path], move_duration=1)
                self.do_reset_recording = True
            


    def set_custom_calibration(self, func):
        """
        Set custom calibration function.
        
        :param func: custom calibration function.
        """
        
        self.update_calibration = types.MethodType(func, self, tobii_controller)


    def use_default_calibration(self):
        """
        Revert calibration function to default one.
        """
        
        self.update_calibration = self.update_calibration_default


    def get_calibration_keymap(self):
        """
        Get current key mapping for selecting calibration points as a dict object.
        """
        
        return self.key_index_dict.copy()


    def set_calibration_keymap(self, keymap):
        """
        Set key mapping for selecting calibration points.
        
        :param dict keymap: Dict object that holds calibration keymap.
            Key of the dict object correspond to PsychoPy key name.
            Value is index of the list of calibration points.
            For example, if you have only two calibration points and 
            want to select these points by 'z' and 'x' key, set keymap
            {'z':0, 'x':1}.
        """
        
        self.key_index_dict = keymap.copy()


    def use_default_calibration_keymap(self):
        """
        Set default key mapping for selecting calibration points.
        """
        
        self.key_index_dict = self.default_key_index_dict.copy()


    def set_calibration_param(self, param_dict):
        """
        Set calibration parameters.
        
        :param dict param_dict: Dict object that holds calibration parameters.
            Use :func:`~psychopy_tobii_controller.tobii_controller.get_calibration_param`
            to get dict object.
        """
        self.calibration_target_dot_size = param_dict['dot_size']
        self.calibration_target_dot.lineColor = param_dict['dot_line_color']
        self.calibration_target_dot.fillColor = param_dict['dot_fill_color']
        self.calibration_target_dot.lineWidth = param_dict['dot_line_width']
        self.calibration_target_disc_size = param_dict['disc_size']
        self.calibration_target_disc.lineColor = param_dict['disc_line_color']
        self.calibration_target_disc.fillColor = param_dict['disc_fill_color']
        self.calibration_target_disc.lineWidth = param_dict['disc_line_width']


    def get_calibration_param(self):
        """
        Get calibration parameters as a dict object.
        The dict object has following keys.

        - 'dot_size': size of the center dot of calibration target.
        - 'dot_line_color': line color of the center dot of calibration target.
        - 'dot_fill_color': fill color of the center dot of calibration target.
        - 'dot_line_width': line width of the center dot of calibration target.
        - 'disc_size': size of the surrounding disc of calibration target.
        - 'disc_line_color': line color of the surrounding disc of calibration target
        - 'disc_fill_color': fill color of the surrounding disc of calibration target
        - 'disc_line_width': line width of the surrounding disc of calibration target
        - 'text_color': color of text
        """
        
        param_dict = {'dot_size':self.calibration_target_dot_size,
                      'dot_line_color':self.calibration_target_dot.lineColor,
                      'dot_fill_color':self.calibration_target_dot.fillColor,
                      'dot_line_width':self.calibration_target_dot.lineWidth,
                      'disc_size':self.calibration_target_disc_size,
                      'disc_line_color':self.calibration_target_disc.lineColor,
                      'disc_fill_color':self.calibration_target_disc.fillColor,
                      'disc_line_width':self.calibration_target_disc.lineWidth}
        return param_dict


    def subscribe(self):
        """
        Start recording.
        """
        
        if self.eyetracker is not None:
            self.gaze_data = []
            self.event_data = []
            self.recording = True
            self.eyetracker.subscribe_to(tobii_research.EYETRACKER_GAZE_DATA, self.on_gaze_data)


    def unsubscribe(self):
        """
        Stop recording.
        """
        
        if self.eyetracker is not None:
            self.eyetracker.unsubscribe_from(tobii_research.EYETRACKER_GAZE_DATA)
            self.recording = False
            self.flush_data()
            self.gaze_data = []
            self.event_data = []

    def start_recording(self):
        self.global_gaze_data = []
        self.subscribe_to_data = True
    
    def stop_recording(self):
        self.subscribe_to_data = False
        

    def subscribe_dict(self):
        if self.eyetracker is not None:
            self.global_gaze_data = []
            self.eyetracker.subscribe_to(tobii_research.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)

    def unsubscribe_dict(self):        
        if self.eyetracker is not None:
            self.eyetracker.unsubscribe_from(tobii_research.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        
       
        
    def on_gaze_data(self, gaze_data):
        """
        Callback function used by
        :func:`~psychopy_tobii_controller.tobii_controller.subscribe`
        
        Usually, users don't have to call this method.
        """
        
        t = gaze_data.system_time_stamp
        lx = gaze_data.left_eye.gaze_point.position_on_display_area[0]
        ly = gaze_data.left_eye.gaze_point.position_on_display_area[1]
        lp = gaze_data.left_eye.pupil.diameter
        lv = gaze_data.left_eye.gaze_point.validity
        rx = gaze_data.right_eye.gaze_point.position_on_display_area[0]
        ry = gaze_data.right_eye.gaze_point.position_on_display_area[1]
        rp = gaze_data.right_eye.pupil.diameter
        rv = gaze_data.right_eye.gaze_point.validity
        self.gaze_data.append((t,lx,ly,lp,lv,rx,ry,rp,rv))
        


    def gaze_data_callback(self, gaze_data):        
        try:
            
            
            lp = gaze_data['left_gaze_origin_in_trackbox_coordinate_system']
            lv = gaze_data['left_gaze_origin_validity']
            rp = gaze_data['right_gaze_origin_in_trackbox_coordinate_system']
            rv = gaze_data['right_gaze_origin_validity']
            self.gaze_data_status = (lp, lv, rp, rv)
                
            gaze_data['current_target_point_on_display_area'] = self.current_target
            if self.subscribe_to_data:
                self.global_gaze_data.append(gaze_data)
            
        except:
            print("Error in callback (dict)")

    def get_current_gaze_position(self):
        """
        Get current (i.e. the latest) gaze position as a tuple of
        (left_x, left_y, right_x, right_y).
        Values are numpy.nan if Tobii fails to get gaze position.
        """
        
        if len(self.gaze_data)==0:
            return (np.nan, np.nan, np.nan, np.nan)
        else:
            lxy = self.get_psychopy_pos(self.gaze_data[-1][1:3])
            rxy = self.get_psychopy_pos(self.gaze_data[-1][5:7])
            return (lxy[0],lxy[1],rxy[0],rxy[1])


    def get_current_pupil_size(self):
        """
        Get current (i.e. the latest) pupil size as a tuple of
        (left, right).
        Values are numpy.nan if Tobii fails to get pupil size.
        """
        
        if len(self.gaze_data)==0:
            return (None,None)
        else:
            return (self.gaze_data[-1][3], #lp
                    self.gaze_data[-1][7]) #rp


    def open_datafile(self, filename, embed_events=False):
        """
        Open data file.
        
        :param str filename: Name of data file to be opened.
        :param bool embed_events: If True, event data is 
            embeded in gaze data.  Otherwise, event data is 
            separately output after gaze data.
        """
        
        if self.datafile is not None:
            self.close_datafile()
        
        self.embed_events = embed_events
        self.datafile = open(filename,'w')
        self.datafile.write('Recording date:\t'+datetime.datetime.now().strftime('%Y/%m/%d')+'\n')
        self.datafile.write('Recording time:\t'+datetime.datetime.now().strftime('%H:%M:%S')+'\n')
        self.datafile.write('Recording resolution:\t%d x %d\n' % tuple(self.win.size))
        if embed_events:
            self.datafile.write('Event recording mode:\tEmbedded\n\n')
        else:
            self.datafile.write('Event recording mode:\tSeparated\n\n')


    def close_datafile(self):
        """
        Write data to the data file and close the data file.
        """
        
        if self.datafile != None:
            self.flush_data()
            self.datafile.close()
        
        self.datafile = None


    def record_event(self,event):
        """
        Record events with timestamp.
        
        Note: This method works only during recording.
        
        :param str event: Any string.
        """
        if not self.recording:
            return
        
        self.event_data.append((tobii_research.get_system_time_stamp(), event))


    def flush_data(self):
        """
        Write data to the data file.
        
        Note: This method do nothing during recording.
        """
        
        if self.datafile == None:
            warnings.warn('data file is not set.')
            return
        
        if len(self.gaze_data)==0:
            return
        
        if self.recording:
            return
        
        self.datafile.write('Session Start\n')
        
        if self.embed_events:
            self.datafile.write('\t'.join(['TimeStamp',
                                           'GazePointXLeft',
                                           'GazePointYLeft',
                                           'PupilLeft',
                                           'ValidityLeft',
                                           'GazePointXRight',
                                           'GazePointYRight',
                                           'PupilRight',
                                           'ValidityRight',
                                           'GazePointX',
                                           'GazePointY',
                                           'Event'])+'\n')
        else:
            self.datafile.write('\t'.join(['TimeStamp',
                                           'GazePointXLeft',
                                           'GazePointYLeft',
                                           'PupilLeft',
                                           'ValidityLeft',
                                           'GazePointXRight',
                                           'GazePointYRight',
                                           'PupilRight',
                                           'ValidityRight',
                                           'GazePointX',
                                           'GazePointY'])+'\n')

        format_string = '%.1f\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f'
        
        timestamp_start = self.gaze_data[0][0]
        num_output_events = 0
        
        if self.embed_events:
            for i in range(len(self.gaze_data)):
                if num_output_events < len(self.event_data) and self.event_data[num_output_events][0] < self.gaze_data[i][0]:
                    event_t = self.event_data[num_output_events][0]
                    event_text = self.event_data[num_output_events][1]
                    
                    if i>0:
                        output_data = self.convert_tobii_record(
                            self.interpolate_gaze_data(self.gaze_data[i-1], self.gaze_data[i], event_t),
                            timestamp_start)
                    else:
                        output_data = ((event_t-timestamp_start)/1000.0, np.nan, np.nan, np.nan, 0,
                                       np.nan, np.nan, np.nan, 0, np.nan, np.nan)
                    
                    self.datafile.write(format_string % output_data)
                    self.datafile.write('\t%s\n' % (event_text))

                    num_output_events += 1
                
                self.datafile.write(format_string % self.convert_tobii_record(self.gaze_data[i], timestamp_start))
                self.datafile.write('\t\n')

            # flush remaining events
            if num_output_events < len(self.event_data):
                for e_i in range(num_output_events, len(self.event_data)):
                    event_t = self.event_data[e_i][0]
                    event_text = self.event_data[e_i][1]
                    
                    output_data = ((event_t-timestamp_start)/1000.0, np.nan, np.nan, np.nan, 0,
                                   np.nan, np.nan, np.nan, 0, np.nan, np.nan)
                    self.datafile.write(format_string % output_data)
                    self.datafile.write('\t%s\n' % (event_text))
        else:
            for i in range(len(self.gaze_data)):
                self.datafile.write(format_string % self.convert_tobii_record(self.gaze_data[i], timestamp_start))
                self.datafile.write('\n')
            
            self.datafile.write('TimeStamp\tEvent\n')
            for e in self.event_data:
                self.datafile.write('%.1f\t%s\n' % ((e[0]-timestamp_start)/1000.0, e[1]))
        
        self.datafile.write('Session End\n\n')
        self.datafile.flush()


    def get_psychopy_pos(self, p):
        """
        Convert PsychoPy position to Tobii coordinate system.
        
        :param p: Position (x, y)
        """
        
        p = (p[0], 1-p[1]) #flip vert
        if self.win.units == 'norm':
            return (2*p[0]-1, 2*p[1]-1)
        elif self.win.units == 'height':
            return ((p[0]-0.5)*self.win.size[0]/self.win.size[1], p[1]-0.5)
        
        p_pix = ((p[0]-0.5)*self.win.size[0], (p[1]-0.5)*self.win.size[1])
        if self.win.units == 'pix':
            return p_pix
        elif self.win.units == 'cm':
            return (self.pix2cm(p_pix[0], self.win.monitor), self.pix2cm(p_pix[1], self.win.monitor))
        elif self.win.units == 'deg':
            return (self.pix2deg(p_pix[0], self.win.monitor), self.pix2deg(p_pix[1], self.win.monitor))
        elif self.win.units in ['degFlat', 'degFlatPos']:
            return (self.pix2deg(np.array(p_pix), self.win.monitor, correctFlat=True))
        else:
            raise ValueError('unit ({}) is not supported.'.format(self.win.units))


    def get_tobii_pos(self, p):
        """
        Convert Tobii position to PsychoPy coordinate system.
        
        :param p: Position (x, y)
        """
        
        if self.win.units == 'norm':
            gp = ((p[0]+1)/2, (p[1]+1)/2)
        elif self.win.units == 'height':
            gp = (p[0]*self.win.size[1]/self.win.size[0]+0.5, p[1]+0.5)
        elif self.win.units == 'pix':
            gp = (p[0]/self.win.size[0]+0.5, p[1]/self.win.size[1]+0.5)
        elif self.win.units == 'cm':
            p_pix = (self.cm2pix(p[0], self.win.monitor), self.cm2pix(p[1], self.win.monitor))
            gp = (p_pix[0]/self.win.size[0]+0.5, p_pix[1]/self.win.size[1]+0.5)
        elif self.win.units == 'deg':
            p_pix = (self.deg2pix(p[0], self.win.monitor), self.deg2pix(p[1], self.win.monitor))
            gp = (p_pix[0]/self.win.size[0]+0.5, p_pix[1]/self.win.size[1]+0.5)
        elif self.win.units in ['degFlat', 'degFlatPos']:
            p_pix = (self.deg2pix(np.array(p), self.win.monitor, correctFlat=True))
            gp = (p_pix[0]/self.win.size[0]+0.5, p_pix[1]/self.win.size[1]+0.5)
        else:
            raise ValueError('unit ({}) is not supported'.format(self.win.units))

        return (gp[0], 1-gp[1]) # flip vert

    def convert_tobii_record(self, record, start_time):
        """
        Convert tobii data to output style.
        Usually, users don't have to call this method.
        
        :param record: element of self.gaze_data.
        :param start_time: Tobii's timestamp when recording was started.
        """
    
        lxy = self.get_psychopy_pos(record[1:3])
        rxy = self.get_psychopy_pos(record[5:7])

        if record[4] == 0 and record[8] == 0: #not detected
            ave = (np.nan, np.nan)
        elif record[4] == 0:
            ave = rxy
        elif record[8] == 0:
            ave = lxy
        else:
            ave = ((lxy[0]+rxy[0])/2.0,(lxy[1]+rxy[1])/2.0)
        
        return ((record[0]-start_time)/1000.0,
                lxy[0], lxy[1], record[3], record[4],
                rxy[0], rxy[1], record[7], record[8],
                ave[0], ave[1])

    def interpolate_gaze_data(self, record1, record2, t):
        """
        Interpolate gaze data between record1 and record2.
        Usually, users don't have to call this method.
        
        :param record1: element of self.gaze_data.
        :param record2: element of self.gaze_data.
        :param t: timestamp to calculate interpolation.
        """
        
        w1 = (record2[0]-t)/(record2[0]-record1[0])
        w2 = (t-record1[0])/(record2[0]-record1[0])
        
        #left eye
        if record1[4] == 0 and record2[4] == 0:
            ldata = record1[1:5]
        elif record1[4] == 0:
            ldata = record2[1:5]
        elif record2[4] == 0:
            ldata = record1[1:5]
        else:
            ldata = (w1*record1[1] + w2*record2[1],
                     w1*record1[2] + w2*record2[2],
                     w1*record1[3] + w2*record2[3],
                     1)

        #right eye
        if record1[8] == 0 and record2[8] == 0:
            rdata = record1[5:9]
        elif record1[4] == 0:
            rdata = record2[5:9]
        elif record2[4] == 0:
            rdata = record1[5:9]
        else:
            rdata = (w1*record1[5] + w2*record2[5],
                     w1*record1[6] + w2*record2[6],
                     w1*record1[7] + w2*record2[7],
                     1)

        return (t,) + ldata + rdata
        