# for cx_freeze (deployment)
import pywt._extensions._cwt
from scipy.linalg import _fblas
import scipy.spatial.ckdtree
import sys
# from tensorflow.contrib.framework.python.ops import gen_variable_ops

from src.tracker import Tracker
import tkinter as tk
import warnings
import cv2
import time
import pickle

from keras.models import load_model

from src.common import *

args = {'flag_shape': 0, 'frame_ind': 1, 'is_online_update': False, 
        'run_model': True, 'save_pos': False, 'is_dl': True}
FLAG = args['flag_shape'] # if 1, use template size as feature
RESIZE = (224, 224)
TRACK_ALGORITHM = 'BOOSTING' # Other alternatives are BOOSTING, KCF, TLD, MEDIANFLOW 
N_MAX = 20 # max number of trying to relocate the target object
TEMP = True

# keyboard return value while it was pressed
KEY_CONTINUE = ord(' ')
KEY_ESC = 27
KEY_ADD = ord('a')
KEY_DELETE = ord('d')
KEY_RETARGET = ord('r')
KEY_MODEL = ord('m')
KEY_MOTION = ord('b')
KEY_HELP = ord('h')
# KEY_UPDATE = ord('u')
KEY_LEFT = 2424832 # might different for different machine
KEY_RIGHT = 2555904
KEY_JUMP = ord('j')
KEY_CHANGE = ord('c')
KEY_RAT = ord('z')

# main logic of the tracker
def main(track_alg):
    warnings.filterwarnings('ignore')
    path = get_path()
    beetle_tracker = Tracker(video_path=path, track_alg=track_alg)
    # read video
    # video = skvideo.io.VideoCapture(find_data_file(beetle_tracker._video))
    video = cv2.VideoCapture(find_data_file(beetle_tracker._video))
    
    # if WRITE:
    # out = cv2.VideoWriter("tracked_%s" % beetle_tracker.file_name, beetle_tracker.fourcc, beetle_tracker.fps, (beetle_tracker.resolution[0], beetle_tracker.resolution[1] + 80))
    # exit if video not opend
    if not video.isOpened():

        beetle_tracker.alert('Could not open video: %s \n %s' % (beetle_tracker._video, find_data_file(beetle_tracker._video)))
        sys.exit()
    # store the length of frame and read the first frame
    beetle_tracker._frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = video.read()
    
    # setup up the window and mouse callback
    # cv2.namedWindow(beetle_tracker.window_name)
    cv2.namedWindow(beetle_tracker.window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(beetle_tracker.window_name, beetle_tracker._mouse_ops)
    while True:
        # Read a new frame and wait for a keypress
        video.set(cv2.CAP_PROP_POS_FRAMES, beetle_tracker.count - 1)
        ok, beetle_tracker.frame = video.read()
        beetle_tracker._fix_target = False
        key = cv2.waitKey(1)
        # check if we have reached the end of the video
        if not ok:
            break
        # resize the frame into 960 x 720
        beetle_tracker._init_frame()
        beetle_tracker.detect_rat_contour()
        
        if len(beetle_tracker._roi) > 0:
            beetle_tracker._roi = [convert(a[0], a[1], a[2], a[3]) for a in beetle_tracker._bboxes]
        
        # if this is init mode, let user targets the beetles
        if beetle_tracker._add_box:
            time.sleep(0.2)
            beetle_tracker._add_bboxes()
        if beetle_tracker.count == 2:
            beetle_tracker._start = time.clock()
        
        # run stop model 
        if len(beetle_tracker._bboxes) > 0 and beetle_tracker._run_model:
            if args['is_online_update']:
                beetle_tracker._is_stop, beetle_tracker._stop_obj = beetle_tracker._detector(FLAG, RESIZE, args['is_dl'], args['is_online_update'])
            else:
                beetle_tracker._is_stop, beetle_tracker._stop_obj = beetle_tracker.detect_and_auto_update(FLAG, RESIZE, args['is_dl'], args['is_online_update'], TEMP, N_MAX)

        if beetle_tracker._run_motion:
            beetle_tracker._motion_detector(FLAG, RESIZE, args['is_dl'], args['is_online_update'], TEMP, N_MAX)

        # if 'r' was pressed or stop model return True, enter to retarget mode
        if key == KEY_RETARGET or beetle_tracker._is_stop:

            if len(beetle_tracker._bboxes) > 0:
                beetle_tracker._retarget_bboxes()
            else:
                beetle_tracker._add_bboxes()
        # if 'a' was pressed, enter add boudning box mode
        elif key == KEY_ADD:
            beetle_tracker._add_bboxes()
        # if 'd' was pressed, enter delete boudning box mode
        elif key == KEY_DELETE:
            beetle_tracker._delete_bboxes()
        elif key == KEY_CONTINUE:
            beetle_tracker._pause_frame()
        elif key == KEY_MODEL:
            beetle_tracker._run_model = not beetle_tracker._run_model
        elif key == KEY_MOTION:
            beetle_tracker._run_motion = not beetle_tracker._run_motion
        elif key == KEY_HELP:
            beetle_tracker.help()
        # elif key == KEY_UPDATE:
        #     beetle_tracker._update = not beetle_tracker._update
        elif key == KEY_JUMP:
            beetle_tracker._jump_frame()
        # restart the program
        elif key == KEY_CHANGE:
        	cv2.destroyAllWindows()
        	main(track_alg=TRACK_ALGORITHM)
        # friendly switch on off for detector
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            beetle_tracker.switch(key)
        elif key == KEY_RAT:
            beetle_tracker._show_rat = not beetle_tracker._show_rat
        # otherwise, update bounding boxes from tracker
        else:
            ok, beetle_tracker._bboxes = beetle_tracker.tracker.update(beetle_tracker.frame)
            beetle_tracker._roi = [convert(a[0], a[1], a[2], a[3]) for a in beetle_tracker._bboxes]
        
        if key == KEY_ESC or (cv2.getWindowProperty(beetle_tracker.window_name, 0) < 0):
            # draw current frame
            beetle_tracker._draw_bbox()
            cv2.imshow(beetle_tracker.window_name, beetle_tracker.frame)
            if beetle_tracker._ask_quit():
                break
            else:
                pass
        if ok:
            beetle_tracker.frame = beetle_tracker.orig_col.copy()
            # draw current frame
            beetle_tracker._draw_bbox()
            # append trace and img
            beetle_tracker._append_record()
            # save image inside the bounding boxes
            beetle_tracker._write_bboxes()
            if args['save_pos'] and len(beetle_tracker.object_name) > 0:
                beetle_tracker._save_pos()
            
            # write current frame to output video
            # if WRITE:
            # out.write(beetle_tracker.frame)
            beetle_tracker.count += 1
            beetle_tracker._n_pass_frame += 1
        else:
            break
        # Display result
        cv2.imshow(beetle_tracker.window_name, beetle_tracker.frame)
        beetle_tracker._ask_add_box()
        
    video.release()
    # if WRITE:
    # out.release()
    cv2.destroyAllWindows()
    # if not args['is_dl']:
    #     pickle.dump(beetle_tracker._model, open('model/%s.dat' % args['model_name'], 'wb'))

if __name__ == '__main__':
	main(track_alg=TRACK_ALGORITHM)