
# Usage: python tracker.py

# for cx_freeze (deployment)
import pywt._extensions._cwt
from scipy.linalg import _fblas
import scipy.spatial.ckdtree
# from multiprocessing.context import Process
# import numpy.core.multiarray

import os, cv2, timeit, sys
import numpy as np
import tkinter as tk
from tkinter import ttk
import _tkinter

from collections import namedtuple
from common import * # comman function 
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno, showerror, showwarning

# for stop-tracking-model
from mahotas.features import haralick, zernike
from skimage.feature import hog
import pickle
import xgboost as xgb

# import argparse
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video_path", help="the path of videos")
# ap.add_argument("-m", "--model_name", default = "model", help="model name")
# ap.add_argument("-f", "--flag_shape", default = 0, type=int, help="whether to use shape of image as features")
# ap.add_argument("-c", "--frame_ind", default = 1, type=int, help="the number of frame to start")
# args = vars(ap.parse_args())

args = {'model_name': 'model', 'flag_shape': 0, 'frame_ind': 1, 'new_f': False}

# some basic variable
WINDOW_NAME = 'Burying Beetle Tracker'
FPS, FOURCC = 30, cv2.VideoWriter_fourcc(*'XVID')
FONT = cv2.FONT_HERSHEY_TRIPLEX
COLOR = [(0, 255, 0), (255, 100, 10), (20, 50, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
TXT_COLOR = (0, 255, 255)
MSG_COLOR = (0, 255, 255)
FONT_SIZE_NM = 0.5
FONT_SIZE_MG = 0.45
FONT_SIZE_EMH = 0.6
TRACK_ALGORITHM = 'MIL' # Other alternatives are BOOSTING, KCF, TLK, MEDIANFLOW 
BAR_HEIGHT = 100
RESIZE = (100, 100)
FLAG = args['flag_shape'] # if 1, use template size as feature
WRITE = True # if True, write output video
Rectangle = namedtuple('Rectangle', 'p1 p2')
N_MAX = 100 # max number of trying to relocate the target object

# keyboard return value while it was pressed
KEY_CONTINUE = ord(' ')
KEY_ESC = 27
KEY_ADD = ord('a')
KEY_DELETE = ord('d')
KEY_RETARGET = ord('r')
KEY_MODEL = ord('m')
KEY_MOTION = ord('b')
KEY_UPDATE = ord('u')
KEY_LEFT = 2424832 # might different for different machine
KEY_RIGHT = 2555904
KEY_JUMP = ord('j')

if args['model_name'] == 'model':
    TEMP = True
else:
    TEMP = False

# parameters for incremental training
params = {'objective': 'binary:logistic', 'verbose': False, 
          'eval_metric': ['logloss'], 'max_depth': 3, 'eta': 0.025,
          'gamma': 0.5, 'subsample': 0.5, 'colsample_bytree': 0.5}
PREFIX = 'training'
# dir_create(PREFIX)
# dir_create('%s/pos' % PREFIX)
# dir_create('%s/new_neg' % PREFIX)

# Define tracker class
# keyhandlr, modeldetector
# super().__init__()
class Tracker():
    
    def __init__(self, video_path, fps = None, fourcc = None, window_name = None, track_alg = None, object_name = None):
        
        self._video = video_path
        self.fourcc = fourcc if fourcc else FOURCC
        self.fps = fps if fps else FPS
        self.window_name = window_name if window_name else WINDOW_NAME
        self.track_alg = track_alg if track_alg else TRACK_ALGORITHM
        self.color = COLOR
        self.font = FONT
        self.count = args['frame_ind']# index of frame
        self.orig_gray = None # original grayscale frame
        self.orig_col = None # original BGR frame
        self.frame = None # current frame

        # setup tracker
        self.tracker = cv2.MultiTracker(self.track_alg)
        
        # different mode while tracking
        self._add_box = True # flag of add bounding box mode
        self._retargeting = False # flag of retargeting bounding box mode
        self._delete_box = False # flag of delete bounding box mode
        self._pause = False # flag of pause mode
        
        # initialize tkinter GUI for asking name while add bounding box
        self.object_name = []
        self.deleted_name = []
        self._askname  = None

        # mouse coordinate
        self._roi_pts = []
        self._mv_pt = None
        
        # initial bounding box
        self._bboxes = []
        self._roi = []
        self._init_bbox = []
        self._len_bbox = 0
        
        # index of beetle
        self._n = 0
        self._fix_target = False

        # variable to store, for current frame, whether the beetle is retarget
        self._n_retarget = 0
        
        # load model
        self._run_model = True
        self._update = False
        if os.path.exists('model'):
            # pass
            self._model = pickle.load(open(find_data_file('model/%s.dat' % args['model_name']), 'rb'))
        else:
            pass
        self._stop_obj = None
        self._is_stop = None

        # background subtractor model, for adding potential bounding box
        self._run_motion = True
        self._bs = cv2.createBackgroundSubtractorMOG2()
        self._pot_rect =  []
        
    # mouse callback method
    def _mouse_ops(self, event, x, y, flags, param):
        
        if len(self._roi) != 0:
            in_rect_boolean = [in_rect(self._mv_pt, rect) for rect in self._roi]
            in_rect_click = True in in_rect_boolean
        else:
            in_rect_click = False
        
        # check if the left mouse button was clicked and whether is in drawing mode
        if (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK) and self._retargeting:
            self._roi_pts = [(x, y)] # record the starting (x, y)
        # check if the left mouse button was released and whether is in drawing mode
        elif (event == cv2.EVENT_LBUTTONUP) and self._retargeting:
            self._roi_pts.append((x, y)) # record the ending (x, y) coordinates
            x0, y0, x1, y1 = [x for tup in self._roi_pts for x in tup]
            self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
            roi_pts = self._roi_pts

            # update model
            self._update_model(type='stop')
            self._bboxes[self._n] = (roi_pts[0][0], roi_pts[0][1],  
                                   roi_pts[1][0] - roi_pts[0][0], roi_pts[1][1] - roi_pts[0][1])
            self._update_model(type='continue')
            self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            self._roi_pts = []
        # check if the left mouse button was clicked, and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK) and self._add_box:
            self._roi_pts = [(x, y)] # reset ROI points and record the starting coordinates
        # check if the left mouse button was released and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONUP) and self._add_box:
            self._roi_pts.append((x, y)) # record the ending (x, y) coordinates
            try:
                x0, y0, x1, y1 = [x for tup in self._roi_pts for x in tup]
                self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
            except:
                pass
            
            if x0 == x1 or y0 == y1:
                pass
            else:
                self._init_bbox.append(self._roi_pts)
            self._roi_pts = []
        # current mouse coordinate
        elif event == cv2.EVENT_MOUSEMOVE:
            self._mv_pt = (x, y) 
        # check if the left mouse buttion was double clicked
        elif event == cv2.EVENT_LBUTTONDBLCLK and self._delete_box and in_rect_click:
            self._delete_box = False
        elif event == cv2.EVENT_RBUTTONDBLCLK and self._retargeting and in_rect_click:
            self._n = in_rect_boolean.index(True)
            self._fix_target = True
    
    # initial frame
    def _init_frame(self):

        # self.frame = cv2.resize(self.frame, self.resolution)
        # extend the height of frame
        self.frame = cv2.copyMakeBorder(self.frame, 0, BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=self.color[7])
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # store original gray scale frame
        self.orig_gray = self.frame.copy()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        # store original frame
        self.orig_col = self.frame.copy()

    # previous frame
    def _previous_frame(self):
        video = cv2.VideoCapture(self._video)
        if self.count > 1:
            self.count -= 1
        else:
            print('Already the first frame!')
        # read previous frame and obtain bounding boxes
        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
        _, self.frame = video.read()
        self._init_frame()
        self._read_bboxes()

    # next frame
    def _next_frame(self):
        video = cv2.VideoCapture(self._video)
        if self.count == self._frame_count:
            print('Already the last frame')
        else:
            self.count += 1
        # read next frame and obtain bounding boxes
        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
        _, self.frame = video.read()
        self._init_frame()
        self._read_bboxes()

    # get frame number method
    def _get_frame_num(self, num):

        try:
            self.count = int(num)
            if self.count > self._frame_count or self.count <= 0:
                showerror('Error', 'Frame number should be bigger than 0 and small than %s' %self._frame_count)
            else:
                self.root.destroy()
        except:
            showerror('Error', 'Require integer')

    # jump frame
    def _jump_frame(self):
        video = cv2.VideoCapture(self._video)

        self.root = tk.Tk()
        self.root.wm_title("Enter # Frame")
        self.root.geometry('240x80')

        L1 = tk.Label(self.root)
        L1.pack(side=tk.TOP)
        ent = tk.Entry(L1, bd = 5, text='Enter desired frame number')
        ent.focus_force()
        ent.bind("<Return>",(lambda event: self._get_frame_num(ent.get())))
        ent.pack(side=tk.TOP)
        
        btn = tk.Button(self.root, text='Submit', command=(lambda: self._get_frame_num(ent.get())))
        btn.pack(side=tk.TOP)
        
        self.root.mainloop()

        # read next frame and obtain bounding boxes
        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
        _, self.frame = video.read()
        self._init_frame()
        self._read_bboxes()

    # draw bouding boxed method         
    def _draw_bbox(self):

        # draw potential
        if len(self._pot_rect) > 0:
            for b in self._pot_rect:
                cv2.rectangle(self.frame, b[0], b[1], (255, 255, 255), 2)

        # draw bounding boxes for different condition
        if not (self._delete_box or self._retargeting):
            for i, b in enumerate(self._roi):
                cv2.rectangle(self.frame, b[0], b[1], self.color[i], 2)
                cv2.putText(self.frame, '%s' % (self.object_name[i]), (b[0][0], b[0][1] - 10), self.font, FONT_SIZE_NM, self.color[i], 1)
            if self._add_box:
                if self._mv_pt:
                    cv2.putText(self.frame, 'Add bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, TXT_COLOR, 1)
                if self._len_bbox > 0:
                    cv2.putText(self.frame, 'Draw a rectangle to add new target', (120, int(self.resolution[1] + 25)), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                else:
                    cv2.putText(self.frame, 'Draw a rectangle to start tracking', (120, int(self.resolution[1] + 25)), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            if self._pause:
                cv2.putText(self.frame, 'PAUSE', (int(self.resolution[0]/2.7), int(self.resolution[1]/2)), self.font, 2, TXT_COLOR, 2)
        else:
            for i, b in enumerate(self._roi):
                if in_rect(self._mv_pt, b) and not self._fix_target:
                    thickness = 4
                    font_thick = 1
                    self._n = i
                else:
                    thickness = 2
                    font_thick = 1
                cv2.rectangle(self.frame, b[0], b[1], self.color[i], thickness)
                cv2.putText(self.frame, 'Current retarget object: %s' % np.array(self.object_name)[int(self._n)], (5, 15), self.font, FONT_SIZE_NM, TXT_COLOR, 1)
                cv2.putText(self.frame, '%s' % (self.object_name[i]), (b[0][0], b[0][1] - 10), self.font, 0.45, self.color[i], font_thick)

            if self._delete_box:
                cv2.putText(self.frame, 'Delete bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, self.color[self._n], 1)
                cv2.putText(self.frame, 'Double click the bounding box to delete', (120, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            elif self._retargeting:
                cv2.putText(self.frame, 'Retarget bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, self.color[self._n], 1)
                cv2.putText(self.frame, 'Retarget by drawing a new rectangle', (120, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                if self._is_stop:
                    string = "Detect that there is no beetle in %s!" % (np.array(self.object_name)[self._stop_obj])
                    cv2.putText(self.frame, string, (5,40), self.font, FONT_SIZE_NM, TXT_COLOR, 1)
        
        cv2.putText(self.frame,'# %s/%s' % (int(self.count), int(self._frame_count)), (5, int(self.resolution[1]) + 25), self.font, FONT_SIZE_MG, TXT_COLOR, 1)
        cv2.putText(self.frame,'# object %s' % self._len_bbox, (5, int(self.resolution[1]) + 50), self.font, FONT_SIZE_MG, TXT_COLOR, 1)
        cv2.putText(self.frame,'resolution: %s x %s   FPS: %s   Model is running: %s' % (self.width, self.height, self.fps, self._run_model), (5, int(self.resolution[1]) + 75), self.font, FONT_SIZE_MG, TXT_COLOR, 1)
        cv2.putText(self.frame, 'r (retarget), a (add), d (delete), space (continue/pause), <- (previouse), -> (next), esc (close)', (120, int(self.resolution[1]) + 50), self.font, FONT_SIZE_MG, TXT_COLOR, 1)        

        # draw current labeling box
        if len(self._roi_pts) != 0:

            if not (self._retargeting or self._add_box):
                pass
            else:
                if self._add_box:
                    drw_color = self.color[len(self._bboxes)]
                elif self._retargeting:
                    drw_color = self.color[self._n]

                x0, y0 = self._roi_pts[0]
                x1, y1 = self._mv_pt

                if x0 == x1 or y0 == y1:
                    pass
                else:
                    pt0, pt1 = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
                    cv2.rectangle(self.frame, pt0, pt1, drw_color, 2)
    
    # save img inside the bounding boxes (for training model)
    def _save_pos(self):

        prefix = PREFIX

        # if not os.path.exists('%s/temp' % prefix):
        #     os.makedirs('%s/temp' % prefix)

        if self.count == 1:
            # for f in os.listdir('%s/temp' % prefix):
            #     if f.startswith(self.video_name):
            #         os.remove('%s/temp/' % prefix + f)
            
            for i, b in enumerate(self._bboxes):
                x, y, w, h = b
                x, y, w, h = int(x), int(y), int(w), int(h)
                img_name = "%s_%04d_%02d.png" % (self.video_name, self.count, int(i+1))
                cv2.imwrite('%s/pos/' % prefix + img_name, self.orig_gray[y:(y+h), x:(x+w)])
                neg_samples = [(max(0, int(x + w*rx)), max(0, int(y + w*ry)), w, h) for rx in [0, 0.25, 0.5, 0.75] for ry in [0, 0.25, 0.5, 0.75] if rx != 0 or ry != 0]
                for j, b_temp in enumerate(neg_samples):
                    x, y, w, h = b_temp
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    img_name = "%s_%05d_%02d_%d.png" % (self.video_name, self.count, int(i+1), int(j+1))
                    cv2.imwrite('%s/new_neg/' % prefix + img_name, self.orig_gray[y:(y+h), x:(x+w)])                
        else:
            txt_name = '%s.txt' % self.video_name
            nframe, n_obj, bboxes = eval(getlines(txt_name, self.count - 1))
            roi = [convert(a[0], a[1], a[2], a[3]) for a in bboxes]
            for i, b in enumerate(self._bboxes):
                x, y, w, h = b
                x, y, w, h = int(x), int(y), int(w), int(h)
                    
                img_name = "%s_%04d_%02d.png" % (self.video_name, self.count, int(i+1))
                cv2.imwrite('%s/pos/' % prefix + img_name, self.orig_gray[y:(y+h), x:(x+w)])
                
                neg_samples = [(max(0, int(x + w*rx)), max(0, int(y + w*ry)), w, h) for rx in [0, 0.25, 0.5, 0.75] for ry in [0, 0.25, 0.5, 0.75] if rx != 0 or ry != 0]
                for j, b_temp in enumerate(neg_samples):
                    x, y, w, h = b_temp
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    img_name = "%s_%05d_%02d_%d.png" % (self.video_name, self.count, int(i+1), int(j+1))
                    cv2.imwrite('%s/new_neg/' % prefix + img_name, self.orig_gray[y:(y+h), x:(x+w)])     

    # read x, y, width, height of bouding boxes of previous frame
    def _read_bboxes(self):

        txt_name = '%s.txt' % self.video_name
        nframe, n_obj, bboxes, self.object_name = eval(getlines(txt_name, self.count - 1))   

        self._bboxes = np.array(bboxes)
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
        self._len_bbox = n_obj
        self._initialize_tracker()

    # write x, y, width, height of bounding boxes to a txt file 
    def _write_bboxes(self):

        txt_name = '%s.txt' % self.video_name
        if os.path.isfile(txt_name):
            file_len = len(open(txt_name, 'r').readlines())
        else:
            file_len = 0

        with open(txt_name, 'a') as f:
            line = '[%s, %s, %s, %s]\n' % (self.count, len(self._bboxes), [list(b) for b in self._bboxes], self.object_name)

            if self.count == (file_len + 1):
                f.write(line)
            else:
            # elif self.count < (file_len + 1):
                with open(txt_name, 'r') as nf:
                    data = nf.readlines()
                    try:
                        data[self.count - 1] = line
                    except:
                        line_temp = ['[%s, %s, [], []]\n' % (c, 0) for c in range(file_len + 1, self.count + 1)]
                        data = data + line_temp
                        data[self.count - 1] = line
                with open(txt_name, 'w') as nf:
                    nf.writelines(data)
        # if there is any rectangle overlapped, save the number of frame
        is_overlapped = any([overlapped(self._roi[i], self._roi[0:i] + self._roi[i+1:]) for i in range(len(self._roi))])
        if is_overlapped:
            txt2_name = 'overlapped_%s.txt' % self.video_name

            with open(txt2_name, 'a') as f:
                line = '%s\n' % self.count
                f.write(line)

    # quit function for string entry
    def _quit_GUI(self, name):

        # append entry to object name
        if name not in self.object_name:
            if name in self.deleted_name:
                self.deleted_name.pop(self.deleted_name.index(name))
            self.object_name.append(name)
            self._askname.destroy()
            # self._askname = None
        else:
            showerror('Error', '%s is already existed' % name)

    # method for typing on combobox list
    def _update_values(self, evt):
        # add entered text to combobox list of values
        widget = evt.widget           # get widget
        txt = widget.get()            # get current text
        vals = widget.cget('values')  # get values
         
        if not vals:
            widget.configure(values = (txt, ))
        elif txt not in vals:
            widget.configure(values = vals + (txt, ))
             
        return 'break'  # don't propagate event

    # after drawing new bounding box, ask user to enter object name
    def _add_name(self):
        self._askname = tk.Tk()
        self._askname.wm_title("Ask name")
        self._askname.geometry('240x80')

        cbp1 = ttk.Labelframe(self._askname, text='Type or choose a name for the object')
        cbp1.pack(side=tk.TOP, fill=tk.BOTH)
        cb = ttk.Combobox(cbp1, values = self.deleted_name)
        if len(self.deleted_name) > 0:
            cb.current(0)
        cb.focus_force()
        cb.bind('<Return>', self._update_values)
        cb.bind("<Return>", lambda event: self._quit_GUI(cb.get()))
        
        cb.pack(side=tk.TOP)

        btn = tk.Button(self._askname, text='Submit', command=(lambda: self._quit_GUI(cb.get())))
        btn.pack(side=tk.TOP)
        
        self._askname.mainloop()

    # ask whether to add potential target
    def _ask_add_box(self):

        if len(self._pot_rect) != 0:    
            self.root = tk.Tk()
            self.root.withdraw()
            if askyesno('Add bounding box', 'Do you wanna add a bouding box?', icon='info'):
                self.root.destroy()
                self._add_bboxes()
            else:
                self.root.destroy()
            self.root.mainloop()

    # ask whether to delele box
    def _ask_delete_box(self):

        self.root = tk.Tk()
        self.root.withdraw()
        result = askyesno('Delete', 'Do you wanna detele %s' % self.object_name[self._n], icon='warning')
        if result:
            self._del_method()
            self.root.destroy()
        else:
            self.root.destroy()
        self.root.mainloop()                    

        return result

    # delete box method
    def _del_method(self):
        # update model if delete a object
        self._update_model()
        # delete selected bounding box
        self._bboxes = np.delete(self._bboxes, self._n, axis=0)
        self._len_bbox -= 1
        self._init_bbox.pop()
        self.deleted_name.append(self.object_name.pop(self._n))
        self._initialize_tracker()
        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

        self._n = 0       

    # show warning
    def alert(self, string='Please at least add one target'):
        self.root = tk.Tk()
        self.root.withdraw()
        showwarning('Alert', string)
        self.root.destroy()
        self.root.mainloop()

    # add bounding boxes
    def _add_bboxes(self):

        self._add_box = True
        if self._add_box:
            # looping the current frame until SPACE was pressed
            while True:
                self.frame = self.orig_col.copy()
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                self._init_bbox = self._roi # update self._init_bbox to current frame ROI
                self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                key_add = cv2.waitKey(1)
                self._draw_bbox()
                cv2.imshow(self.window_name, self.frame)
                if (self._len_bbox != len(self._init_bbox)):

                    self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                    
                    ok = self.tracker.add(self.frame, self._bboxes[self._len_bbox]) # adding new bounding box
                    self._len_bbox += 1
                    self._add_name()
                # break the loop if SPACE was pressed
                elif key_add == KEY_CONTINUE:
                    if self._len_bbox > 0:
                        self._add_box = False
                        self._n_retarget += 1
                        break
                    else:
                        self.alert()
                # if 'r' was pressed, enter retarget boudning box mode
                elif key_add == KEY_RETARGET:
                    self._add_box = False
                    self._retarget_bboxes()
                    break
                # if 'd' was pressed, enter delete boudning box mode
                elif key_add == KEY_DELETE:
                    self._add_box = False
                    self._delete_bboxes()
                    break
                # back to previous frame is LEFT KEY was pressed
                elif key_add == KEY_LEFT:
                    self._previous_frame()
                # go to next frame is RIGHT KEY was pressed
                elif key_add == KEY_RIGHT:
                    self._next_frame()

        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

    # retarget bounding boxes
    def _retarget_bboxes(self):

        video = cv2.VideoCapture(self._video)
        # reset retargeting object index to object 1
        self._n = self._stop_obj[0] if self._is_stop else 0 
        
        self._retargeting = True
        # looping the current frame until SPACE was pressed
        while True:
            key_reset = cv2.waitKey(1)
            cv2.imshow(self.window_name, self.frame)

            # break the loop if SPACE was pressed
            if key_reset == KEY_CONTINUE:
                if self._len_bbox > 0:
                    self._retargeting = False
                    self._n_retarget += 1
                    break
                else:
                    self.alert()
                    self._retargeting = False
                    self._add_bboxes()
                    break
            # back to previous frame is LEFT KEY was pressed
            elif key_reset == KEY_LEFT:
                self._previous_frame()
            # go to next frame is RIGHT KEY was pressed
            elif key_reset == KEY_RIGHT:
                self._next_frame()
            # if 'a' was pressed, enter add boudning box mode
            elif key_reset == KEY_ADD:
                self._retargeting = False
                self._add_bboxes()
                break
            # if 'd' was pressed, enter delete boudning box mode
            elif key_reset == KEY_DELETE:
                self._retargeting = False
                self._delete_bboxes()
                break
            # else just keep looping at current frame
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
                _, self.frame = video.read()
                self._init_frame()
            # reset new ROI and draw bounding boxes
            self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            self._draw_bbox()

        self._initialize_tracker()

    # delete bounding boxes
    def _delete_bboxes(self):
        self._delete_box = True
        while True:
            key_delete = cv2.waitKey(1)
            if not self._delete_box:
                self.frame = self.orig_col.copy()
                
                self._del_method()
                # quit the delete mode
                # self._delete_box = False
                self._delete_box = True
            # if 'r' was pressed, enter retarget boudning box mode
            elif key_delete == KEY_RETARGET:
                self._delete_box = False
                self._retarget_bboxes()
                break
            # if 'a' was pressed, enter add boudning box mode
            elif key_delete == KEY_ADD:
                self._delete_box = False
                self._add_bboxes()
                break
            # break the loop if SPACE was pressed
            elif key_delete == KEY_CONTINUE:
                if self._len_bbox > 0:
                    self._delete_box = False
                    break
                else:
                    self.alert()
                    self._delete_box = False
                    self._add_bboxes()
                    break
            # back to previous frame is LEFT KEY was pressed
            elif key_delete == KEY_LEFT:
                self._previous_frame()
            # go to next frame is RIGHT KEY was pressed
            elif key_delete == KEY_RIGHT:
                self._next_frame()
            else:
                self.frame = self.orig_col.copy()

            self._draw_bbox()
            cv2.imshow(self.window_name, self.frame)
    
    # pause video
    def _pause_frame(self): 
        self._pause = True
        while True:
            key_pause = cv2.waitKey(1)
            # break the loop if SPACE was pressed
            if key_pause == KEY_CONTINUE:
                if self._len_bbox > 0:
                    self._pause = False
                    break
                else:
                    self.alert()
                    self._pause = False
                    self._add_bboxes()
                    break
            # if 'r' was pressed, enter retarget boudning box mode
            elif key_pause == KEY_RETARGET:
                self._pause = False
                self._retarget_bboxes()
                break
            # if 'a' was pressed, enter add boudning box mode
            elif key_pause == KEY_ADD:
                self._pause = False
                self._add_bboxes()
                break
            # if 'd' was pressed, enter delete boudning box mode
            elif key_pause == KEY_DELETE:
                self._pause = False
                self._delete_bboxes()
                break
            # back to previous frame is LEFT KEY was pressed
            elif key_pause == KEY_LEFT:
                self._previous_frame()
            # go to next frame is RIGHT KEY was pressed
            elif key_pause == KEY_RIGHT:
                self._next_frame()
            elif key_pause == KEY_JUMP:
                self._jump_frame()
            else:
                self.frame = self.orig_col.copy()

            self._draw_bbox()
            cv2.imshow(self.window_name, self.frame)

    # initialize multi tracker
    def _initialize_tracker(self):
        self.tracker = cv2.MultiTracker(self.track_alg)
        self.tracker.add(self.frame, tuple(self._bboxes))

    # extract features for stop model
    @staticmethod
    def extract_features(img, flag):
        orig = img.shape
        img = cv2.resize(img, RESIZE)

        f_hog = hog(cv2.resize(img, (64, 64)), orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1))
        f_lbs = lbs.describe(img)
        
        # normalized intensity histogram
        hist = cv2.calcHist([img], [0], None, [256], (0, 256))
        n_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX).flatten()
        # HuMoments
        HuMoments = cv2.HuMoments(cv2.moments(img)).flatten()
        # Zernike moments
        Zernike = zernike(img, radius=1, degree=8)
        # Haralick
        Haralick = haralick(img).mean(axis=0)

        if flag == 1:
            return np.array(list(n_hist) + list(HuMoments) + list(Zernike) + list(Haralick) + list(np.array(orig)))
        elif flag == 0:
            if args['new_f']:
                return np.array(list(n_hist) + list(HuMoments) + list(Zernike) + list(Haralick) + list(f_hog) + list(f_lbs))    
            else:
                return np.array(list(n_hist) + list(HuMoments) + list(Zernike) + list(Haralick))
            
    # detect motion for potential target
    def _motion_detector(self):

        fg_mask = self._bs.apply(self.orig_gray.copy())

        potential_rect = []

        th = cv2.threshold(fg_mask.copy(), 230, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('hi', th)
        # cv2.waitKey(1)
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 5)), iterations=2)
        
        # get contours from dilated frame
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # get estimated bounding box of contour
            x, y, w, h = cv2.boundingRect(c)
            # calculate the area
            area = cv2.contourArea(c)
            if area > 200 and area < 1000:
                potential_rect.append((x, y, w, h))

        if len(potential_rect) > 0:
            X_potential = np.array([self.extract_features(self.orig_gray.copy()[y:(y+h), x:(x+w)], FLAG) for x, y, w, h in potential_rect])
            pred_potential = self._model.predict(xgb.DMatrix(X_potential))
            pred_potential = np.array([1 if v < 0.5 else 0 for v in pred_potential])

            if any(pred_potential == 1):
                self._pot_rect = np.array(potential_rect)[np.where(pred_potential == 1)]
                self._pot_rect = list(self._pot_rect)
                pot_rect_orig = self._pot_rect

                for i, r in enumerate(self._pot_rect):
                    x, y, w, h = r
                    is_obj_prop = 0
                    n_try = 0    
                    while is_obj_prop < 0.7:
                        x1, y1, w1, h1 = random_target(r, 30, 2, True)
                        if (w1 / h1) > 2 or (h1 / w1) > 2:
                            pass
                        else:
                            random_roi_feature = [self.extract_features(self.orig_gray[y1:(y1+h1), x1:(x1+w1)], FLAG)]
                            random_roi_feature = np.array(random_roi_feature)

                            if not TEMP:
                                pred = self._model.predict_proba(random_roi_feature)
                                is_obj_prop = pred[0][0]
                            else:
                                pred = self._model.predict(xgb.DMatrix(random_roi_feature))
                                is_obj_prop = 1 - pred[0]
                        n_try += 1
                        if n_try >= N_MAX / 2:
                            break
                    if n_try < N_MAX:        
                        r = (x1, y1, w1, h1)
                    else:
                        try:
                            pot_rect_orig.pop(pot_rect_orig.index(r))
                        except:
                            pass

                self._pot_rect = pot_rect_orig

                self._pot_rect = [convert(a[0], a[1], a[2], a[3]) for a in self._pot_rect]
                filter_condition = [not overlapped(rect, self._roi) for rect in self._pot_rect]
                self._pot_rect = [rect for (rect, v) in zip(self._pot_rect, filter_condition) if v]


            else:
                self._pot_rect = []
        else:
            self._pot_rect =  []

    # stop model and auto update with random ROI which has highest probability contain beetle
    def _stop_to_continue_update(self):

        X = []
        stop_obj = []

        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            x, y, w, h = int(x), int(y), int(w), int(h)
            img = self.orig_gray[y:(y+h), x:(x+w)]
            X.append(self.extract_features(img, FLAG))
        
        X = np.array(X)
        if not TEMP: 
            pred = self._model.predict_proba(X)
            pred = np.array([1 if v[1] > 0.4 else 0 for v in pred]) 
        else:
            pred = self._model.predict(xgb.DMatrix(X))
            pred = np.array([1 if v > 0.4 else 0 for v in pred]) # i.e. continue only if the probability of beetle being in bounding box > 0.6
        
        is_overlapped = np.array([overlapped(self._roi[i], self._roi[0:i] + self._roi[i+1:]) for i in range(len(self._roi))])
        
        for i, v in enumerate(is_overlapped):
            if v:
                stop_obj.append(i)

        pred[is_overlapped] = 0

        # if any bounding box has value 1, resampling candidates of bounding box
        if any(pred == 1):
            
            start = timeit.default_timer()
            
            for bbox_ind in np.where(pred == 1)[0]:
                continue_prop = 0
                n_try = 0    
                while continue_prop < 0.65:
                    x, y, w, h = random_target(self._bboxes[bbox_ind])

                    if (w / h) > 2 or (h / w) > 2 or area(Rectangle(self._roi[bbox_ind][0], self._roi[bbox_ind][1])) / area(Rectangle((x, y), (x+w, y+h))) < 0.7:
                        pass
                    else:
                        print(x, y, w, h)

                        random_roi_feature = [self.extract_features(self.orig_gray[y:(y+h), x:(x+w)], FLAG)]
                        random_roi_feature = np.array(random_roi_feature)

                        if not TEMP:
                            pred = self._model.predict_proba(random_roi_feature)
                            continue_prop = pred[0][0]
                        else:
                            pred = self._model.predict(xgb.DMatrix(random_roi_feature))
                            continue_prop = 1 - pred[0]

                        print(continue_prop)
                        n_try += 1
                        if n_try >= N_MAX:
                            self._n = bbox_ind
                            is_deleted = self._ask_delete_box()
                            break
                        else:
                            is_deleted = False

                if not is_deleted:
                    # reset bounding box with highest probability
                    if n_try != N_MAX:
                        self._bboxes[bbox_ind] = (x, y, w, h)
                    # delete bounding box if the number of randomly retargeting exceeds 5000 times
                    else:
                        stop_obj.append(bbox_ind)

                    print('Done retargeting %s...' % (self.object_name[bbox_ind]))

            print('Retargeting take %s secs', round(timeit.default_timer() - start, 2))

            if len(stop_obj) == 0:
                self._initialize_tracker()
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

                return False, None
            else:
                # pass
                return True, stop_obj
        else:
            return False, None

    # update model
    def _update_model(self, type='stop'):
        if self._update:            
            x, y, w, h = self._bboxes[self._n]
            img = self.orig_gray[y:(y+h), x:(x+w)]
            if type == 'stop':
                xg_train = xgb.DMatrix(np.array([self.extract_features(img, FLAG)]), np.array([1]))
            else:
                xg_train = xgb.DMatrix(np.array([self.extract_features(img, FLAG)]), np.array([0]))

            self._model = xgb.train(params, xg_train, 50, xgb_model = self._model)
            print('Model updated')

    # main logic of the tracker
    def start(self):
        
        # read video
        video = cv2.VideoCapture(self._video)
        self.width = int(video.get(3))
        self.height = int(video.get(4))
        self.fps = int(video.get(5))
        self.alert(str(self.fps))
        self.resolution = (self.width, self.height)
        self.file_name = self._video.split('/')[-1]
        self.video_name = self.file_name.split('.')[0]

        if WRITE:
            out = cv2.VideoWriter("tracked_%s" % self.file_name, self.fourcc, self.fps, (self.resolution[0], self.resolution[1] + 80))
        # exit if video not opend
        if not video.isOpened():

            self.alert('Could not open video: %s \n %s' % (self._video, find_data_file(self._video)))
            sys.exit()
        # store the length of frame and read the first frame
        self._frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = video.read()
        
        # setup up the window and mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self._mouse_ops)
        
        while True:
            # Read a new frame and wait for a keypress
            video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
            ok, self.frame = video.read()
            self._fix_target = False
            key = cv2.waitKey(1)
            # check if we have reached the end of the video
            if not ok:
                break
            # resize the frame into 960 x 720
            self._init_frame()
            
            if len(self._roi) > 0:
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            
            # if this is init mode, let user targets the beetles
            if self._add_box:
                self._add_bboxes()
            
            # run stop model 
            if len(self._bboxes) > 0 and self._run_model:
                self._is_stop, self._stop_obj = self._stop_to_continue_update()

            self._motion_detector()

            # if 'r' was pressed or stop model return True, enter to retarget mode
            if key == KEY_RETARGET or self._is_stop:
                if len(self._bboxes) > 0:
                    self._retarget_bboxes()
                else:
                    self._add_bboxes()
            # if 'a' was pressed, enter add boudning box mode
            elif key == KEY_ADD:
                self._add_bboxes()
            # if 'd' was pressed, enter delete boudning box mode
            elif key == KEY_DELETE:
                self._delete_bboxes()
            elif key == KEY_CONTINUE:
                self._pause_frame()
            elif key == KEY_MODEL:
                self._run_model = not self._run_model
            elif key == KEY_MOTION:
                self._run_motion = not self._run_motion
            elif key == KEY_UPDATE:
                self._update = not self._update
            # otherwise, update bounding boxes from tracker
            else:
                ok, self._bboxes = self.tracker.update(self.frame)
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            
            if ok:
                self.frame = self.orig_col.copy()
                # draw current frame
                self._draw_bbox()
                # save image inside the bounding boxes
                self._write_bboxes()
                # self._save_pos()
                # write current frame to output video
                if WRITE:
                    out.write(self.frame)
                self.count += 1
            # Display result
            cv2.imshow(self.window_name, self.frame)

            self._ask_add_box()

            if key == KEY_ESC: break
            
        video.release()
        if WRITE:
            out.release()
        cv2.destroyAllWindows()
        pickle.dump(self._model, open('model/%s.dat' % args['model_name'], 'wb'))

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    path = askopenfilename(title='Where is the video?', filetypes=[('video file (*.avi;*.mp4)','*.avi;*.mp4')])
    root.destroy()
    root.mainloop()
    # path = args['video_path']
    burying_beetle_tracker = Tracker(video_path=path, fps=15)
    burying_beetle_tracker.start()