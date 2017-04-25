# Usage: python tracker.py

import os, cv2, timeit, sys
import numpy as np
from tkinter.filedialog import askopenfilename
# for stop-tracking-model
from mahotas.features import haralick, zernike
from skimage.feature import hog
import pickle

# some basic variable
WINDOW_NAME = 'Burying Beetle Tracker'
OBJECT_NAME = 'beetle'
FPS, FOURCC, RESOLUTION = 30, cv2.VideoWriter_fourcc(*'XVID'), (960, 720)
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
         (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
TRACK_ALGORITHM = 'MIL' # Other alternatives are BOOSTING, KCF, TLK, MEDIANFLOW 
RESIZE = (118, 88)

# keyboard value
KEY_CONTINUE = ord(' ')
KEY_LEFT = 2424832 # might different for different machine
KEY_RIGHT = 2555904

# some function
# convert (x0, y0, width, height) into ((x0, y0), (x1, y1)), where x1 = x0 + width; y1 = y0 + height
convert = lambda x, y, w, h: ((int(x), int(y)), (int(x + w), int(y + h))) 
# add randomness to an integer
vary = lambda x, var: x + np.random.randint(-var, var)
# random a new bounding box with a bounding box as input
def random_target(bbox, var = 10):
    x, y, w, h = bbox
    x1, y1, w1, h1 = vary(x, var), vary(y, var), vary(w, var), vary(h, var)
    
    return x1, y1, w1, h1
# get specific line from a text file
def getlines(txt, n_line):
    with open(txt, 'r') as f:
        lines = f.readlines()

    return lines[n_line]

# Define tracker class
class Tracker:
    
    def __init__(self, video_path, fps = None, fourcc = None, window_name = None, track_alg = None, object_name = None):
        
        self._video = video_path
        self.fourcc = fourcc if fourcc else FOURCC
        self.fps = fps if fps else FPS
        self.resolution = RESOLUTION
        self.window_name = window_name if window_name else WINDOW_NAME
        self.object_name = object_name if object_name else OBJECT_NAME
        self.track_alg = track_alg if track_alg else TRACK_ALGORITHM
        self.color = COLOR
        self.font = FONT
        self.count = 1  # index of frame
        self.orig_gray = None # original grayscale frame
        self.orig_col = None # original BGR frame
        self.frame = None # current frame

        # setup tracker
        self.tracker = cv2.MultiTracker(self.track_alg)
        
        # different mode while tracking
        self._init = True # flag of initialize mode or add bounding box mode
        self._drawing = False # flag of drawing mode
        self._delete_box = False # flag of delete bounding box mode
        
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

        # variable to store, for current frame, whether the beetle is retarget
        self._n_retarget = 0
        
        # load model
        if os.path.exists('model'):
            self._model = pickle.load(open('model/beetle_resize_118_88_no_shape.dat', 'rb'))
        else:
            self._model = model
        
    # mouse callback method
    def _mouse_ops(self, event, x, y, flags, param):
        
        # check if the left mouse button was clicked and whether is in drawing mode
        if (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK) and self._drawing:
            self._roi_pts = [(x, y)] # record the starting (x, y)
        # check if the left mouse button was released and whether is in drawing mode
        elif (event == cv2.EVENT_LBUTTONUP) and self._drawing:
            self._roi_pts.append((x, y)) # record the ending (x, y) coordinates
            x0, y0, x1, y1 = [x for tup in self._roi_pts for x in tup]
            self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
            roi_pts = self._roi_pts
            self._bboxes[self._n] = (roi_pts[0][0], roi_pts[0][1],  
                                   roi_pts[1][0] - roi_pts[0][0], roi_pts[1][1] - roi_pts[0][1])
            self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            self._roi_pts = []
        # check if the left mouse button was clicked, and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK) and self._init:
            self._roi_pts = [(x, y)] # reset ROI points and record the starting coordinates
            
        # check if the left mouse button was released and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONUP) and self._init:
            self._roi_pts.append((x, y)) # record the ending (x, y) coordinates
            x0, y0, x1, y1 = [x for tup in self._roi_pts for x in tup]
            self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
            
            if x0 == x1 or y0 == y1:
                pass
            else:
                self._init_bbox.append(self._roi_pts)
            self._roi_pts = []
        # current mouse coordinate
        elif event == cv2.EVENT_MOUSEMOVE:
            self._mv_pt = (x, y) 

        # scroll up or down to select index of object
        elif event == cv2.EVENT_MOUSEWHEEL and (self._drawing or self._delete_box):
            if flags < 0:
                if self._n == 0:
                    pass
                else:
                    self._n -= 1
                print('Current retarget object: %s' % int(self._n + 1))
            else:
                if self._n == len(self._bboxes) - 1:
                    pass
                else:
                    self._n += 1
                print('Current retarget object: %s' % int(self._n + 1))    
    
    # draw bouding boxed method
    def _draw_bbox(self):
        
        # draw bounding boxes
        for i, b in enumerate(self._roi):
            cv2.rectangle(self.frame, b[0], b[1], self.color[i], 2)
            cv2.putText(self.frame, '%s %s' % (self.object_name, int(i+1)), (b[0][0], b[0][1] - 10), self.font, 0.45, self.color[i], 2)
        
        cv2.putText(self.frame,'# %s' %int(self.count), (10,20), self.font, 0.7, (0,255,255), 1)
        cv2.putText(self.frame,'# retarget %s' %self._n_retarget, (100, 20), self.font, 0.7, (0,255,255), 1)

        # draw current labeling box
        if len(self._roi_pts) != 0:

            if not (self._drawing or self._init):
                pass
            else:
                if self._init:
                    drw_color = self.color[len(self._bboxes)]
                elif self._drawing:
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

        if self.count == 1:
            for f in os.listdir('pos'):
                if f.startswith(self.video_name):
                    os.remove('pos/' + f)
        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            img_name = "%s_%04d_%02d.png" % (self.video_name, self.count, int(i+1))
            cv2.imwrite('pos/' + img_name, self.orig_gray[y:(y+h), x:(x+w)])

    # write x, y, width, height of bounding boxes to a txt file 
    def _write_bboxes(self):

        txt_name = '%s.txt' % self.video_name
        if os.path.isfile(txt_name):
            file_len = len(open(txt_name, 'r').readlines())
        else:
            file_len = 0

        with open(txt_name, 'a') as f:
            line = '[%s, %s, %s]\n' % (self.count, len(self._bboxes), [list(b) for b in self._bboxes])

            if self.count == (file_len + 1):
                f.write(line)
            elif self.count < (file_len + 1):
                with open(txt_name, 'r') as nf:
                    data = nf.readlines()
                    data[self.count - 1] = line
                with open(txt_name, 'w') as nf:
                    nf.writelines(data)
            else:
                print('This is strange!')
                pass

    # add bounding box
    def _add_bbox(self):

        self._init = True
        # update self._init_bbox to current frame ROI
        self._init_bbox = self._roi
        if self._init:
            # looping the current frame until SPACE was pressed
            while True:
                self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                key_add = cv2.waitKey(1)
                self._draw_bbox()
                cv2.putText(self.frame, 'Draw rectangle to add new target and press SPACE to continue', (10,50), self.font, 0.8, (51,153,255), 1)
                cv2.imshow(self.window_name, self.frame)
                if (self._len_bbox != len(self._init_bbox)):
                    self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                    ok = self.tracker.add(self.frame, self._bboxes[self._len_bbox]) # adding new bounding box
                    self._len_bbox += 1

                elif key_add == KEY_CONTINUE:
                    self._init = False
                    break
        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

    # delete bounding boxes
    def _delete_bboxes(self):
        self._n = 0
        self._delete_box = True
        # self.frame = self._draw_bbox(self.frame)
        while True:
            key_delete = cv2.waitKey(1)

            if key_delete == KEY_CONTINUE:
                self._delete_box = False
                break
            else:
                self.frame = self.orig_col.copy()

            self._draw_bbox()
            cv2.putText(self.frame, 'Current retarget object: %s' % int(self._n + 1), (270,20), self.font, 0.8, (0,255,255), 1)
            cv2.putText(self.frame, 'Press SPACE to delete and continue tracking', (10,50), self.font, 0.8, (153,255,51), 1)
            cv2.imshow(self.window_name, self.frame)
        # delete selected bounding box
        self._bboxes = np.delete(self._bboxes, self._n, axis=0)
        self._len_bbox -= 1
        self._init_bbox.pop()

        # reinitialize multi tracker
        self.tracker = cv2.MultiTracker(self.track_alg)
        self.tracker.add(self.frame, tuple(self._bboxes))
        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

    # read x, y, width, height of bouding boxes of previous frame
    def _read_bboxes(self):

        txt_name = '%s.txt' % self.video_name
        nframe, n_obj, bboxes = eval(getlines(txt_name, self.count - 1))

        assert nframe == self.count
        self._bboxes = np.array(bboxes)

    # extract features for stop model
    @staticmethod
    def extract_features(img):
        orig = img.shape
        img = cv2.resize(img, RESIZE)
        # normalized intensity histogram
        hist = cv2.calcHist([img], [0], None, [256], (0, 256))
        n_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX).flatten()
        # HuMoments
        HuMoments = cv2.HuMoments(cv2.moments(img)).flatten()
        # Zernike moments
        Zernike = zernike(img, radius=1, degree=8)
        # Haralick
        Haralick = haralick(img).mean(axis=0)

        return np.array(list(n_hist) + list(HuMoments) + list(Zernike) + list(Haralick))
    
    # stop model
    def _stop_or_continue(self):
        
        X = [] # initial input for model
        
        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            img = self.orig_gray[y:(y+h), x:(x+w)]
            X.append(self.extract_features(img))
        
        X = np.array(X)
        pred = self._model.predict(X)

        if any(pred == 1):
            
            return True, np.where(pred == 1)[0]
        else:

            return False, None

    # stop model and auto update with random ROI which has highest probability contain beetle
    def _stop_to_continue_update(self):

        X = []

        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            img = self.orig_gray[y:(y+h), x:(x+w)]
            X.append(self.extract_features(img))
        
        X = np.array(X)
        pred = self._model.predict_proba(X)

        prediction = np.array([0 if v[0] > 0.6 else 1 for v in pred])
        stop = any(prediction == 1)
        
        if stop:
            self._n_retarget += 1
            string1 = "Detect there is no beetle in bounding box %s" % (np.where(prediction == 1)[0] + 1)
            string2 = "Relocalizing target object..."
            cv2.putText(self.frame, string1, (10,60), self.font, 0.8, (0,255,255), 1)
            cv2.putText(self.frame, string2, (10,120), self.font, 0.8, (0,255,255), 1)

            n_random = 20
            for bbox_ind in np.where(prediction == 1)[0]:
                random_roi = [random_target(self._bboxes[bbox_ind]) for _ in range(n_random)]

                random_roi_feature = [self.extract_features(self.orig_gray[x:(x+w), y:(y+h)]) for x, y, w, h in random_roi]
                random_roi_feature = np.array(random_roi_feature)

                pred = self._model.predict_proba(random_roi_feature)
                continue_prop = [p[0] for p in pred]

                most_likely_target = continue_prop.index(max(continue_prop))

                self._bboxes[bbox_ind] = np.array(random_roi)[most_likely_target]
                print('Done retargeting beetle %s' % bbox_ind)

                # reinitialize multi tracker
            self.tracker = cv2.MultiTracker(self.track_alg)
            self.tracker.add(self.orig_col, tuple(self._bboxes))
            self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

    # main logic of the tracker
    def start(self):
        
        # read video
        video = cv2.VideoCapture(self._video)
        self.file_name = self._video.split('\\')[-1]
        self.video_name = self.file_name.split('.')[0]
        out = cv2.VideoWriter("tracked_%s" % self.file_name, self.fourcc, self.fps, self.resolution)
        # exit if video not opend
        if not video.isOpened():
            print('Could not open video')
            sys.exit()
        # store the length of frame and read the first frame
        self._frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = video.read()
        
        # setup up the window and mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_ops)
        print('Instructions: \n')
        print('Draw a rectange around the target objects using left mouse button. ')
        print('Press space to start tracking. ')
        print('After tracking begins,')
        print('Press "r" key to retarget;')
        print('Press "a" key to add bounding box;')
        print('Press "d" key to delete bounding box. \n')
        
        while True:
            # Read a new frame and wait for a keypress
            video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
            ok, self.frame = video.read()
            key = cv2.waitKey(1)
            # check if we have reached the end of the video
            if not ok:
                break
            # resize the frame into 960 x 720
            self.frame = cv2.resize(self.frame, self.resolution)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.orig_gray = self.frame.copy()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            self.orig_col = self.frame
            
            if len(self._roi) != 0:
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            
            # if this is init mode, let user targets the beetles
            if self._init:

                while True:
                    key_init = cv2.waitKey(1)
                    self.frame = self.orig_col.copy()
                    self._draw_bbox()
                    cv2.imshow(self.window_name, self.frame)
                    
                    if len(self._init_bbox) != 0:
                        self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                    
                    self._draw_bbox()
                    if (self._len_bbox != len(self._init_bbox)):
                        ok = self.tracker.add(self.frame, self._bboxes[self._len_bbox])
                        self._len_bbox += 1

                    elif key_init == KEY_CONTINUE:
                        self._init = False
                        break
            
            is_stop, stop_obj = self._stop_or_continue()
            # self._stop_to_continue_update()

            # if 'r' was pressed or stop model return True, enter to retarget mode
            if key == ord('r') or is_stop:
            # if key == ord('r'):
                # reset retargeting object index to object 1
                self._n = stop_obj[0]
                self._n_retarget += 1
                print('Scroll the mouse wheel to choose index of object that needs retargeting \n')
                print('Current retarget object: %s' % int(self._n + 1))
               
                self._drawing = True
                # looping the current frame until SPACE was pressed
                while True:
                    key_reset = cv2.waitKey(1)
                    cv2.imshow(self.window_name, self.frame)

                    # break the loop if SPACE was pressed
                    if key_reset == KEY_CONTINUE:
                        self._drawing = False
                        break
                    # back to previous frame is LEFT KEY was pressed
                    elif key_reset == KEY_LEFT:
                        if self.count > 1:
                            self.count -= 1
                        else:
                            print('Already the first frame!')
                        # read previous frame and obtain bounding boxes
                        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
                        _, self.frame = video.read()
                        self._read_bboxes()
                    # go to next frame is RIGHT KEY was pressed
                    elif key_reset == KEY_RIGHT:
                        if self.count == self._frame_count:
                            print('Already the last frame')
                        else:
                            self.count += 1
                        # read next frame and obtain bounding boxes
                        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
                        _, self.frame = video.read()
                        self._read_bboxes()
                    # else just keep looping at current frame
                    else:
                        video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
                        _, self.frame = video.read()
                        cv2.putText(self.frame, 'Current retarget object: %s' % int(self._n + 1), (270,20), self.font, 0.8, (0,255,255), 1)
                        cv2.putText(self.frame, 'Retarget by drawing new rectangle and press SPACE to continue', (10,80), self.font, 0.8, (0,255,255), 1)
                        if is_stop:
                            string = string = "Detect that there is no beetle in bounding box %s" %(stop_obj + 1)
                            cv2.putText(self.frame, string, (10,50), self.font, 0.8, (0,255,255), 1)
                    # reset new ROI and draw bounding boxes
                    self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                    self._draw_bbox()

                # reinitialize multi tracker
                self.tracker = cv2.MultiTracker(self.track_alg)
                self.tracker.add(self.frame, tuple(self._bboxes))
            # if 'a' was pressed, entering add boudning box mode
            elif key == ord('a'):
                print('Draw rectangle using left mouse button to add new target object')
                self._add_bbox()
            # if 'd' was pressed, entering delete boudning box mode
            elif key == ord('d'):
                print('Scroll the mouse wheel to choose index of object that needs deleting (default is %s 1) \n' % (self.object_name))
                self._delete_bboxes()
            # otherwise, update bounding boxes from tracker
            else:
                ok, self._bboxes = self.tracker.update(self.frame)
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
            
            if ok:
                # draw current frame
                # self.frame = self._draw_bbox(self.frame)
                self._draw_bbox()
                # save image inside the bounding boxes
                self._save_pos()
                self._write_bboxes()
                # write current frame to output video
                out.write(self.frame)
                self.count += 1
            # Display result
            cv2.imshow(self.window_name, self.frame)
            
            if key == 27: break
            
        video.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path = askopenfilename(title='Where is the video?')
    # path = sys.argv[1]
    burying_beetle_tracker = Tracker(video_path=path, fps=3)
    burying_beetle_tracker.start()
    exit()