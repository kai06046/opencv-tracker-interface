import cv2, time
from src.common import *
import tkinter as tk
from tkinter.messagebox import askyesno, askokcancel, showerror, showwarning, showinfo
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
KEY_HELP = ord('h')

BAR_HEIGHT = 130
TXT_COLOR = (0, 255, 255)
MSG_COLOR = (0, 255, 255)
WHITE = (255, 255, 255)
FONT_SIZE_NM = 0.45
FONT_SIZE_MG = 0.55
FONT_SIZE_EMH = 0.75

PREFIX = 'training'

class BasicOperation(object):

    # save img inside the bounding boxes (for training model)
    def _save_pos(self):

        prefix = PREFIX
        ratio = self._ratio
        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            x, y, w, h = int(x), int(y), int(w), int(h)
            img_name = "%s_%02d_%04d.png" % (self.video_name, int(i+1), self.count)
            img  = self.orig_gray[y:(y+h), x:(x+w)]

            try:
                # diff = compare_images(img, self._record[self.object_name[i]][-1])
                # print('diff between current and the last frame of beetle %s: %s' % (self.object_name[i], diff))

                # if diff > 0.03:
                cv2.imwrite(('%s/beetle_pos/' % prefix) + img_name, img)
                neg_samples = [(max(0, int(x + w*rx*xsign)), max(0, int(y + h*ry*ysign)), w, h) for rx in ratio for ry in ratio for xsign in [-1, 1] for ysign in [-1, 1] if rx != 0 or ry != 0]
                for j, b_temp in enumerate(neg_samples):
                    try:
                        x, y, w, h = b_temp
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        img_name = "%s_%05d_%02d_%d.png" % (self.video_name, self.count, int(i+1), int(j+1))
                        cv2.imwrite(('%s/new_neg/' % prefix) + img_name, self.orig_gray[y:(y+h), x:(x+w)]) 
                    except:
                        print('pass boundary')
            except Exception as e:
                print(e)

    # read x, y, width, height of bouding boxes of previous frame
    def _read_bboxes(self):

        txt_name = '%s.txt' % self.video_name
        try:
            nframe, n_obj, bboxes, self.object_name = eval(getlines(txt_name, self.count - 1))   
        except:
            file_len = len(open(txt_name, 'r').readlines())
            with open(txt_name, 'r') as nf:
                data = nf.readlines()
                line_temp = ['[%s, %s, [], []]\n' % (c, 0) for c in range(file_len + 1, self.count + 1)]
                data = data + line_temp
            with open(txt_name, 'w') as nf:
                nf.writelines(data)
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

    # initialize multi tracker
    def _initialize_tracker(self):
        self.tracker = cv2.MultiTracker(self.track_alg)
        self.tracker.add(self.frame, tuple(self._bboxes))

    # append meta data of tracking
    def _append_record(self):
        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            x, y, w, h = int(x), int(y), int(w), int(h)
            img = self.orig_gray[y:(y+h), x:(x+w)].copy()
            if self.object_name[i] in self._record.keys():
                self._record[self.object_name[i]]['image'].append(img)
                self._record[self.object_name[i]]['trace'].append((x, y))
            else:
                self._record[self.object_name[i]] = {'image': [img], 'trace':[(x, y)], 'detect': True}

class KeyHandler(BasicOperation):

    # initial frame
    def _init_frame(self):

        # extend the height of frame
        self.frame = cv2.copyMakeBorder(self.frame, 0, BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=self.color[7])
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # store original gray scale frame
        self.orig_gray = self.frame.copy()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        # store original frame
        self.orig_col = self.frame.copy()

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
            if y1 <= self.height and y0 <= self.height:
                self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
                roi_pts = self._roi_pts

                # update model (closed this method currently)
                # self._update_model(type='stop')
                self._bboxes[self._n] = (roi_pts[0][0], roi_pts[0][1],  
                                       roi_pts[1][0] - roi_pts[0][0], roi_pts[1][1] - roi_pts[0][1])
                # self._update_model(type='continue')
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                if self.object_name[self._n] in self._record.keys(): self._record[self.object_name[self._n]]['trace'] = []
            else:
                self.alert('Please draw bounding box in side the frame')
                self._roi_pts = []
            self._roi_pts = []
        # check if the left mouse button was clicked, and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONDOWN) and self._add_box:
            self._roi_pts = [(x, y)] # reset ROI points and record the starting coordinates
        # check if the left mouse button was released and whether is in init mode
        elif (event == cv2.EVENT_LBUTTONUP) and self._add_box:
            self._roi_pts.append((x, y)) # record the ending (x, y) coordinates
            try:
                x0, y0, x1, y1 = [x for tup in self._roi_pts for x in tup]
                if y1 <= self.height and y0 <= self.height:
                    self._roi_pts[0], self._roi_pts[1] = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
                    if x0 == x1 or y0 == y1:
                        pass
                    else:
                        self._init_bbox.append(self._roi_pts)
                else:
                    self.alert('Please draw bounding box in side the frame')
                    self._roi_pts = []
            except:
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
    def _jump_frame(self, action_func=None):
        action_func = action_func if action_func else self._pause_frame

        video = cv2.VideoCapture(self._video)
        orig_frame_count = self.count

        self.root = tk.Tk()
        self.root.wm_title("Enter # Frame")
        self.root.geometry('240x80')

        center(self.root)
        L1 = tk.Label(self.root)
        L1.pack(side=tk.TOP)
        ent = tk.Entry(L1, bd = 5, text='Enter desired frame number')
        ent.focus_force()
        ent.bind("<Return>",(lambda event: self._get_frame_num(ent.get())))
        ent.pack(side=tk.TOP)
        
        btn = tk.Button(self.root, text='Submit', command=(lambda: self._get_frame_num(ent.get())))
        btn.pack(side=tk.TOP)

        self.root.mainloop()

        self._record = {}
        self._stop_obj = None
        self._is_stop = False

        if self.count != orig_frame_count:
            # read next frame and obtain bounding boxes
            video.set(cv2.CAP_PROP_POS_FRAMES, self.count - 1)
            _, self.frame = video.read()
            self._init_frame()
            self._read_bboxes()
            if self._len_bbox != 0 or self._add_box:
                action_func()
            else:
                self.alert('There is no bounding box to be retargeted')
                self._retargeting = False
                self._add_bboxes()

    # switch of model
    def switch(self, key):
        try:
            if key == ord('1'):
                self._record[self.object_name[0]]['detect'] = not self._record[self.object_name[0]]['detect']
            elif key == ord('2'):
                self._record[self.object_name[1]]['detect'] = not self._record[self.object_name[1]]['detect']
            elif key == ord('3'):
                self._record[self.object_name[2]]['detect'] = not self._record[self.object_name[2]]['detect']
            elif key == ord('4'):
                self._record[self.object_name[3]]['detect'] = not self._record[self.object_name[3]]['detect']
        except:
            print('No rules for %s' % key)

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

                if key_add == KEY_ESC or (cv2.getWindowProperty(self.window_name, 0) < 0):
                    # draw current frame
                    self._draw_bbox()
                    cv2.imshow(self.window_name, self.frame)
                    if self._ask_quit():
                        self.out.release()
                        exit()
                    else:
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
                        cv2.setMouseCallback(self.window_name, self._mouse_ops)

                cv2.imshow(self.window_name, self.frame)
                if (self._len_bbox != len(self._init_bbox)):

                    self._add_name()
                    self._roi_pts = []
                    if len(self.object_name) == self._len_bbox + 1:
                        self._bboxes = [(r[0][0], r[0][1], r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in self._init_bbox]
                        ok = self.tracker.add(self.frame, self._bboxes[self._len_bbox]) # adding new bounding box
                        self._len_bbox += 1
                    
                # break the loop if SPACE was pressed
                elif key_add == KEY_CONTINUE:

                    if self._len_bbox > 0:
                        self._add_box = False
                        break
                    else:
                        if self._ask_quit('Confirm', 'Do you want to continue without any bounding box?'):
                            self._add_box = False
                            break
                        else:
                            pass
                # if 'r' was pressed, enter retarget boudning box mode
                elif key_add == KEY_RETARGET:
                    if self._len_bbox != 0:
                        self._add_box = False
                        self._retarget_bboxes()
                        break
                    else:
                        self.alert('There is no bounding box to be retargeted')
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
                elif key_add == KEY_JUMP:
                    temp_count = self.count
                    self._jump_frame(self._add_bboxes)
                    if temp_count != self.count:
                        self._add_box = False
                        break
                elif key_add == KEY_MODEL:
                    self._run_model = not self._run_model
                elif key_add == KEY_MOTION:
                    self._run_motion = not self._run_motion
                elif key_add == KEY_HELP:
                    self.help()
                else:
                    pass

        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

    # retarget bounding boxes
    def _retarget_bboxes(self):

        video = cv2.VideoCapture(self._video)
        # reset retargeting object index to object 1
        if self._stop_obj:
            self._n = self._stop_obj[0] if len(self._stop_obj) > 0 else 0 
        else:
            self._n = 0
        
        self._retargeting = True
        self._roi_pts = []

        # looping the current frame until SPACE was pressed
        while True:

            key_reset = cv2.waitKey(1)
            if key_reset == KEY_ESC or (cv2.getWindowProperty(self.window_name, 0) < 0):
                # draw current frame
                self._draw_bbox()
                cv2.imshow(self.window_name, self.frame)
                if self._ask_quit():
                    self.out.release()
                    exit()
                else:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
                    cv2.setMouseCallback(self.window_name, self._mouse_ops)
            cv2.imshow(self.window_name, self.frame)

            # break the loop if SPACE was pressed
            if key_reset == KEY_CONTINUE:
                if self._len_bbox > 0:
                    self._retargeting = False
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
            elif key_reset == KEY_JUMP:
                temp_count = self.count
                self._jump_frame(self._retarget_bboxes)
                if temp_count != self.count:
                    self._retargeting = False
                    break
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
            elif key_reset == KEY_MODEL:
                self._run_model = not self._run_model
            elif key_reset == KEY_MOTION:
                self._run_motion = not self._run_motion
            elif key_reset == KEY_HELP:
                self.help()                
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
                if self._len_bbox != 0:
                    self._delete_box = False
                    self._retarget_bboxes()
                    break
                else:
                    self.alert('There is no bounding box to be retargeted')   
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
            elif key_delete == KEY_JUMP:
                temp_count = self.count
                self._jump_frame(self._delete_bboxes)
                if temp_count != self.count:
                    self._delete_box = False
                    break
            elif key_delete == KEY_MODEL:
                self._run_model = not self._run_model
            elif key_delete == KEY_MOTION:
                self._run_motion = not self._run_motion
            elif key_delete == KEY_HELP:
                self.help()
            elif key_delete == KEY_ESC or (cv2.getWindowProperty(self.window_name, 0) < 0):
                # draw current frame
                self._draw_bbox()
                cv2.imshow(self.window_name, self.frame)
                if self._ask_quit():
                    self.out.release()
                    exit()
                else:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
                    cv2.setMouseCallback(self.window_name, self._mouse_ops)
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
                if self._len_bbox != 0:
                    self._pause = False
                    self._retarget_bboxes()
                    break
                else:
                    self.alert('There is no bounding box to be retargeted')
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
                temp_count = self.count
                self._jump_frame(self._pause_frame)
                if temp_count != self.count:
                    break
            elif key_pause == KEY_MODEL:
                self._run_model = not self._run_model
            elif key_pause == KEY_MOTION:
                self._run_motion = not self._run_motion
            elif key_pause == KEY_HELP:
                self.help()
            elif key_pause == KEY_ESC or (cv2.getWindowProperty(self.window_name, 0) < 0):
                # draw current frame
                self._draw_bbox()
                cv2.imshow(self.window_name, self.frame)
                if self._ask_quit():
                    self.out.release()
                    exit()
                else:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
                    cv2.setMouseCallback(self.window_name, self._mouse_ops)
            else:
                self.frame = self.orig_col.copy()

            self._draw_bbox()
            cv2.imshow(self.window_name, self.frame)

    # draw bouding boxed method         
    def _draw_bbox(self):

        # draw initial status
        cv2.putText(self.frame, 'TRACKING', (5, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, WHITE, 1)
        cv2.putText(self.frame, 'ADD', (155, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, WHITE, 1)
        cv2.putText(self.frame, 'PAUSE', (230, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, WHITE, 1)
        cv2.putText(self.frame, 'RETARGET', (340, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, WHITE, 1)
        cv2.putText(self.frame, 'DELETE', (495, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, WHITE, 1)
        cv2.putText(self.frame, 'AUTOADD', (615, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR if self._run_motion else WHITE, 1)

        # draw potential bounding box that has target object
        if len(self._pot_rect) > 0:
            for b in self._pot_rect:
                x, y, w, h = b
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # draw bounding boxes for different condition
        if not (self._delete_box or self._retargeting):
            
            if self._add_box:
                if self._mv_pt:
                    cv2.putText(self.frame, 'Add bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, TXT_COLOR, 1)
                if self._len_bbox > 0:
                    cv2.putText(self.frame, 'Draw a rectangle to add new target', (120, int(self.resolution[1] + 75)), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                else:
                    cv2.putText(self.frame, 'Draw a rectangle to start tracking', (120, int(self.resolution[1] + 75)), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                # change status color
                cv2.putText(self.frame, 'ADD', (155, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            elif self._pause:
                # change status color
                cv2.putText(self.frame, 'PAUSE', (230, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            else:
                # change status color
                cv2.putText(self.frame, 'TRACKING', (5, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            for i, b in enumerate(self._roi):
                cv2.rectangle(self.frame, b[0], b[1], self.color[i], 2)
                cv2.putText(self.frame, '%s' % (self.object_name[i]), (b[0][0], b[0][1] - 10), self.font, FONT_SIZE_NM, self.color[i], 1)
        else:
            if self._delete_box:
                cv2.putText(self.frame, 'Delete bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, self.color[self._n], 1)
                cv2.putText(self.frame, 'Double click the bounding box to delete', (120, int(self.resolution[1]) + 75), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                # change status color
                cv2.putText(self.frame, 'DELETE', (495, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
            elif self._retargeting:
                cv2.putText(self.frame, 'Retarget bounding box', (self._mv_pt[0], self._mv_pt[1] + 5), self.font, FONT_SIZE_NM, self.color[self._n], 1)
                cv2.putText(self.frame, 'Retarget by drawing a new rectangle', (120, int(self.resolution[1]) + 75), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                # change status color
                cv2.putText(self.frame, 'RETARGET', (340, int(self.resolution[1]) + 25), self.font, FONT_SIZE_EMH, MSG_COLOR, 1)
                if self._is_stop:
                    string = "Detect that there is no beetle in %s!" % (np.array(self.object_name)[self._stop_obj])
                    cv2.putText(self.frame, string, (5,40), self.font, FONT_SIZE_NM, TXT_COLOR, 1)
            # change bounding box thickness while mouse is inside it
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

        # draw the switch of detector for each object
        if len(self.object_name) > 0:
            for i, name in enumerate(self.object_name):
                try:
                    c = self.color[i] if self._record[name]['detect'] else WHITE
                except:
                    c = self.color[i]
                x, y, w, h = 5 + i * 125, int(self.resolution[1]) + 33, 100, 20
                rect = np.array( [[[x, y],[x+w,y],[x+w,y+h],[x,y+h]]], dtype=np.int32 )
                cv2.fillPoly(self.frame, rect, c)
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.color[i], 3)
                
                # cv2.putText(self.frame, name, (5 + i * 30, int(self.resolution[1]) + 50), self.font, FONT_SIZE_EMH * 1.2, c, 1)
        else:
            cv2.putText(self.frame, 'NO OBJECT', (5, int(self.resolution[1] + 50)), self.font, FONT_SIZE_EMH, WHITE, 1)

        # draw basic information
        cv2.putText(self.frame,'# %s/%s' % (int(self.count), int(self._frame_count)), (5, int(self.resolution[1]) + 75), self.font, FONT_SIZE_MG, TXT_COLOR, 1)
        cv2.putText(self.frame,'# object %s' % self._len_bbox, (5, int(self.resolution[1]) + 100), self.font, FONT_SIZE_MG, TXT_COLOR, 1)
        cv2.putText(self.frame,'resolution: %s x %s   FPS: %s   Press h to view the setting'% (self.width, self.height, (round(self._n_pass_frame/(time.clock() - self._start), 3) if self._start else 0)), (120, int(self.resolution[1]) + 100), self.font, FONT_SIZE_MG, TXT_COLOR, 1)

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