from src.common import *
import cv2
import warnings
import time
import matplotlib.path as mplPath

from collections import namedtuple

# for multithread
from multiprocessing.dummy import Pool as ThreadPool

Rectangle = namedtuple('Rectangle', 'p1 p2')
KEY_ESC = 27

pool = ThreadPool(4)

class BeetleDetector(object):
    # extract features for stop model
    @staticmethod
    def extract_features(img, inputShape):        
        img = cv2.resize(img, inputShape, interpolation = cv2.INTER_NEAREST)
        img = img / 255.0
        return img
    
    # stop model and auto update with random ROI which has highest probability contain beetle
    def detect_and_auto_update(self, inputShape, N_MAX):

        X = []
        stop_obj = []

        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            x, y, w, h = int(x), int(y), int(w), int(h)
            img = self.orig_col[y:(y+h), x:(x+w)]
            X.append(self.extract_features(img, inputShape))
        
        X = np.array(X)
        pred = self._model.predict(X)
        print('Probability of bounding box has no beetle in frame %s' % self.count)
        for i, p in enumerate(pred):
            print('%s: %s' % (self.object_name[i], round(pred[i][0], 3)))

        prediction = np.array([1 if v[0] > 0.4 else 0 for v in pred]) # i.e. continue only if the probability of bounding box has beetle < 0.4                
        
        is_overlapped = np.array([overlapped(self._roi[i], self._roi[0:i] + self._roi[i+1:]) for i in range(len(self._roi))])
        
        if any(is_overlapped): print('Detect overlapped')

        # for i, v in enumerate(is_overlapped):
        #     if v:
        #         stop_obj.append(i)

        # prediction[is_overlapped] = 0

        # if any bounding box has value 1, resampling candidates of bounding box
        if any(prediction == 1):
            
            start = time.clock()
            
            temp = self.frame.copy()

            for bbox_ind in np.where(prediction == 1)[0]:
                
                orig_prob = pred[bbox_ind][0] # current bounding box index background probability

                # switch on/off for detector
                if self.object_name[bbox_ind] in self._record.keys():
                    if self._record[self.object_name[bbox_ind]]['detect']:
                        continue_prop = 0
                    else:
                        continue_prop = 1
                        is_retarget = False
                        x, y, w, h = self._bboxes[bbox_ind]
                else:
                    continue_prop = 0

                try:
                    trace = np.array(self._record[self.object_name[bbox_ind]]['trace'])
                    
                    # last 10 diff
                    trace_diff = np.vstack((np.diff(trace, axis=0)[::-1][:10], trace[-1] - trace[-min(10, len(trace))])).tolist()
                except:
                    trace_diff = []
                n_try = 1    

                key = cv2.waitKey(1)
                while continue_prop < 0.65:
                    if key == KEY_ESC or (cv2.getWindowProperty(self.window_name, 0) < 0):
                        # draw current frame
                        self._draw_bbox()
                        cv2.imshow(self.window_name, self.frame)
                        if self._ask_quit():
                            beetle_tracker.out.release()
                            exit()
                        else:
                            cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
                            cv2.setMouseCallback(self.window_name, self._mouse_ops)
                            cv2.imshow(self.window_name, self.frame)                    
                    
                    # section for using group rectangle
                    if len(trace_diff) > 0:
                        x, y, w, h = self._bboxes[bbox_ind]
                        temp_diff = trace_diff.pop(0)
                        x, y, w, h = max(0, int(x + temp_diff[0])), max(0, int(y + temp_diff[1])), int(w), int(h)
                        ratio_cond = '(w / h) > 1.8 or (h / w) > 1.8'
                        boundary_cond = 'y > self.height or (y+h) > self.height or x > self.width or (x+w) > self.width'
                        if  eval(ratio_cond) or eval(boundary_cond) or area(Rectangle(self._roi[bbox_ind][0], self._roi[bbox_ind][1])) / area(Rectangle((x, y), (x+w, y+h))) < 0.7:
                            pass
                        else:
                            # display random bounding box
                            cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.color[bbox_ind], 2)
                            cv2.putText(self.frame, '# retargeting of %s: %s/%s' % (self.object_name[bbox_ind], n_try, N_MAX), (20, 35), self.font, 1, (0, 255, 255), 2)
                            self._draw_bbox()
                            cv2.imshow(self.window_name, self.frame)
                            key_model = cv2.waitKey(1)

                            if key_model == 27:
                                print('break from the auto retargeting by trace')
                                is_retarget = False
                                break

                            random_roi_feature = [self.extract_features(self.orig_col[y:(y+h), x:(x+w)], inputShape)]
                            random_roi_feature = np.array(random_roi_feature)
                            pred_random = self._model.predict(random_roi_feature)[0][0]

                            continue_prop = 1- pred_random
                    else:
                        n_candidates = 30
                        x, y, w, h = self._bboxes[bbox_ind]
                        
                        if (w / float(h)) > 2.5:
                            print('manual change')
                            h = int(h * 1.5)
                        elif (h / float(w)) > 2.5:
                            print('manual change')
                            w = int(w * 1.5)

                        random_candidates = random_target((x, y, w, h), size=(n_candidates, 1)).astype('int')
                        gp_rects, _ = cv2.groupRectangles(random_candidates.tolist(), min(2, len(random_candidates) - 1), eps=1)
                        gp_rects.astype('int')
                        x, y, w, h = gp_rects[0]
                        cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.color[bbox_ind], 2)
                        cv2.putText(self.frame, '# retargeting of %s: %s/%s' % (self.object_name[bbox_ind], n_try, N_MAX), (20, 35), self.font, 1, (0, 255, 255), 2)
                        self._draw_bbox()
                        cv2.imshow(self.window_name, self.frame)
                        key_model = cv2.waitKey(1)
                        
                        if key_model == 27:
                            print('break from the auto retargeting')
                            is_retarget = False
                            break

                        random_roi_feature = [np.expand_dims(self.extract_features(self.orig_col[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])], inputShape), 0) for r in random_candidates]

                        pred_time = time.clock()
                        print('predicting...')
                        pred_random = pool.map(self._model.predict, random_roi_feature)
                        pred_random = np.vstack(tuple(pred_random))
                        print('predict %s candidates took %s secs' % (n_candidates, round(time.clock() - pred_time)))
                        thres = np.where(pred_random < 0.5)[0]
                        len_thres = len(thres)
                        if len_thres > 0:
                            
                            if len_thres > 1:
                                print('merging %s candidates...' % len_thres)
                                rects = random_candidates[thres]
                                gp_rects, _ = cv2.groupRectangles(rects.tolist(), min(2, len(rects) - 1), eps=1)
                                gp_rects.astype('int')
                                if len(gp_rects) > 0:
                                    x, y, w, h = gp_rects[0]
                                else:
                                    x, y, w, h = rects[0]
                            else:
                                x, y, w, h = random_candidates[thres][0]

                            continue_prop = 1- pred_random[thres].mean()
                        else:
                            print('no candidates has beetle...')
                            continue_prop = 0

                    # section end for using group rectangle
                    print('Probability of beetle in bounding box %s: %s' % (self.object_name[bbox_ind], round(continue_prop, 3)))
                    n_try += 1
                    if n_try >= N_MAX:
                        break
                    else:
                        is_retarget = False
                    self.frame = temp.copy()

                if not is_retarget:
                    # reset bounding box with highest probability
                    if n_try != N_MAX:
                        self._bboxes[bbox_ind] = [int(x), int(y), int(w), int(h)]
                        print('Done auto retargeting %s...' % (self.object_name[bbox_ind]))
                    # delete bounding box if the number of randomly retargeting exceeds 5000 times
                    else:
                        stop_obj.append(bbox_ind)
                        print('Please help retarget %s...' % (self.object_name[bbox_ind]))

            print('Retargeting take %s secs', round(time.clock() - start, 2))

            if len(stop_obj) == 0:
                self._initialize_tracker()
                self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

                return False, None
            else:
                # pass
                return True, stop_obj
        else:
            return False, None

class MotionDetector(object):
    # detect motion for potential target
    def _motion_detector(self, inputShape, N_MAX):

        fg_mask = self._bs.apply(self.orig_gray.copy())

        potential_rect = []

        th = cv2.threshold(fg_mask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        # get contours from dilated frame
        _, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # get estimated bounding box of contour
            x, y, w, h = cv2.boundingRect(c)
            # calculate the area
            area = cv2.contourArea(c)
            if area > 180 and area < 800 and x < 1200 and (w/h < 2) and (h/w < 2):
                if area < 350:
                    x = int(x / 1.03)
                    y = int(y / 1.03)
                    w = int(w * 2.2)
                    h = int(h * 1.6)

                potential_rect.append((x, y, w, h))

        potential_rect = [convert(a[0], a[1], a[2], a[3]) for a in potential_rect]
        filter_condition = [not overlapped(rect, self._roi) for rect in potential_rect]
        potential_rect = [rect for (rect, v) in zip(potential_rect, filter_condition) if v]
        potential_rect = [(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in potential_rect]

        if len(potential_rect) > 0:
            
            X_potential = np.array([self.extract_features(self.orig_col.copy()[y:(y+h), x:(x+w)], inputShape) for x, y, w, h in potential_rect])
            
            pred_potential_prob = self._model.predict(X_potential)
            print('potential has no beetle probability: %s' % pred_potential_prob)
            pred_potential = np.array([1 if v[0] < 0.35 else 0 for v in pred_potential_prob])

            if any(pred_potential == 1):
                self._pot_rect = np.array(potential_rect)[np.where(pred_potential == 1)]
                self._pot_rect = self._pot_rect.tolist()
            else:
                self._pot_rect = []
        else:
            self._pot_rect =  []

class RatDetector(object):

    def detect_rat_contour(self):

        blurred = cv2.GaussianBlur(self.orig_gray, (5, 5), 0)

        # Otsu's thresholding
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # T = otsu(blurred)
        # th = self.orig_gray.copy()
        # th[th > T] = 255
        # th[th < 255] = 0
        _, cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # find contour with the biggest area
        self.rat_cnt = sorted(cnts, key=cv2.contourArea)[-1]

    def detect_on_rat(self, bbox):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        try:
            cnt = self.rat_cnt.reshape(len(self.rat_cnt), 2)
            poly = mplPath.Path(cnt)
            on_rat = False
            for x in [x1, x2]:
                for y in [y1, y2]:
                    on_rat = on_rat or poly.contains_point((x, y))
        except Exception as e:
            print('Error in detect_on_rat method', e)
            on_rat = False
        return on_rat