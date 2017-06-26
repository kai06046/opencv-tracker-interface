from src.common import *
import xgboost as xgb

from mahotas.features import haralick, zernike
from skimage.feature import hog
import cv2
import warnings
import time

from collections import namedtuple

# for multithread
from multiprocessing.dummy import Pool as ThreadPool

# parameters for incremental training
params = {'objective': 'binary:logistic', 'verbose': False, 
          'eval_metric': ['logloss'], 'max_depth': 3, 'eta': 0.025,
          'gamma': 0.5, 'subsample': 0.5, 'colsample_bytree': 0.5}

Rectangle = namedtuple('Rectangle', 'p1 p2')
KEY_ESC = 27

pool = ThreadPool(4)

class BeetleDetector(object):
    # extract features for stop model
    @staticmethod
    def extract_features(img, flag, inputShape, is_dl, is_olupdate):
        if is_dl:
            # img_to_array(load_img)
            img = cv2.resize(img, inputShape, interpolation = cv2.INTER_NEAREST)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            # preprocess(image)
            # print('preprocess image took %s secs' % (round(time.clock() - start, 2)))
            return img
        else:
            orig = img.shape
            if not is_olupdate:
                img = cv2.resize(img, inputShape)
            warnings.filterwarnings('ignore')
            f_hog = hog(cv2.resize(img, (64, 64)), orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1))
            f_lbs = lbs.describe(img)
            
            # normalized intensity histogram
            hist = cv2.calcHist([img], [0], None, [32], (0, 256))
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
                return np.array(list(n_hist) + list(HuMoments) + list(Zernike) + list(Haralick) + list(f_hog) + list(f_lbs))    
    
    # stop model and auto update with random ROI which has highest probability contain beetle
    def detect_and_auto_update(self, flag, inputShape, is_dl, is_olupdate, TEMP, N_MAX):

        X = []
        stop_obj = []

        for i, b in enumerate(self._bboxes):
            x, y, w, h = b
            x, y, w, h = int(x), int(y), int(w), int(h)
            if is_dl:
                img = self.orig_col[y:(y+h), x:(x+w)]
            else:    
                img = self.orig_gray[y:(y+h), x:(x+w)]
            X.append(self.extract_features(img, flag, inputShape, is_dl, is_olupdate))
        
        if is_dl:
            X = np.array(X)
            pred = self._model.predict(X)
            print('Probability of bounding box has no beetle in frame %s' % self.count)
            for i, p in enumerate(pred):
                print('%s: %s' % (self.object_name[i], round(pred[i][0], 3)))

            prediction = np.array([1 if v[0] > 0.4 else 0 for v in pred]) # i.e. continue only if the probability of bounding box has beetle < 0.4
        else:
            X = np.array(X)
            if not TEMP: 
                pred = self._model.predict_proba(X)
                pred = np.array([1 if v[1] > 0.4 else 0 for v in pred]) 
            else:
                pred = self._model.predict(xgb.DMatrix(X)) # probability of there is no beetle in bounding box
                print('Probability of bouding box has no beetle: %s' % pred)
                pred = np.array([1 if v > 0.5 else 0 for v in pred]) # i.e. continue only if the probability of beetle being in bounding box > 0.6
                
        
        is_overlapped = np.array([overlapped(self._roi[i], self._roi[0:i] + self._roi[i+1:]) for i in range(len(self._roi))])
        
        if any(is_overlapped): print('Detect overlapped')

        # for i, v in enumerate(is_overlapped):
        #     if v:
        #         stop_obj.append(i)

        prediction[is_overlapped] = 0

        # if any bounding box has value 1, resampling candidates of bounding box
        if any(prediction == 1):
            
            start = time.clock()
            
            temp = self.frame.copy()

            for bbox_ind in np.where(prediction == 1)[0]:
                orig_prob = pred[bbox_ind][0] # current bounding box index background probability

                continue_prop = 0
                try:
                    trace = np.array(self._record[self.object_name[bbox_ind]]['trace'])
                    
                    # last 10 diff
                    trace_diff = np.vstack((np.diff(trace, axis=0)[::-1][:10], trace[-1] - trace[0])).tolist()
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
                                print('break from the auto retargeting')
                                is_retarget = False
                                break
                                # return False, None

                            if is_dl:
                                random_roi_feature = [self.extract_features(self.orig_col[y:(y+h), x:(x+w)], flag, inputShape, is_dl, is_olupdate)]
                                random_roi_feature = np.array(random_roi_feature)
                                pred_random = self._model.predict(random_roi_feature)[0][0]
                                # if pred_random < orig_prob:
                                    # print('original prob: %s pred_random: %s' % (orig_prob, pred_random))
                                    # orig_prob = pred_random
                                    # self._bboxes[bbox_ind] = (x, y, w, h)
                                    # print('update bbox but still continue update')

                                continue_prop = 1- pred_random
                            else:
                                random_roi_feature = [self.extract_features(self.orig_gray[y:(y+h), x:(x+w)], flag, inputShape, is_dl, is_olupdate)]
                                random_roi_feature = np.array(random_roi_feature)
                                if not TEMP:
                                    pred = self._model.predict_proba(random_roi_feature)
                                    continue_prop = pred[0][0]
                                else:
                                    pred = self._model.predict(xgb.DMatrix(random_roi_feature))
                                    continue_prop = 1 - pred[0]
                    else:
                        n_candidates = 30
                        x, y, w, h = self._bboxes[bbox_ind]
                        
                        if (w / float(h)) > 2.5:
                            print('manual change')
                            h = h * 1.5
                        elif (h / float(w)) > 2.5:
                            print('manual change')
                            w = w * 1.5                        

                        random_candidates = random_target_a((x, y, w, h), size=(n_candidates, 1)).astype('int')
                        gp_rects, _ = cv2.groupRectangles(random_candidates.tolist(), min(2, len(random_candidates) - 1), eps=1)
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

                        random_roi_feature = [np.expand_dims(self.extract_features(self.orig_col[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])], flag, inputShape, is_dl, is_olupdate), 0) for r in random_candidates]
                        # random_roi_feature = np.array(random_roi_feature)

                        pred_time = time.clock()
                        print('predicting...')
                        pred_random = pool.map(self._model.predict, random_roi_feature)
                        pred_random = np.vstack(tuple(pred_random))
                        # pred_random = self._model.predict(random_roi_feature)
                        print('predict %s candidates took %s secs' % (n_candidates, round(time.clock() - pred_time)))
                        thres = np.where(pred_random < 0.5)[0]
                        len_thres = len(thres)
                        if len_thres > 0:
                            
                            if len_thres > 1:
                                print('merging %s candidates...' % len_thres)
                                rects = random_candidates[thres]
                                gp_rects, _ = cv2.groupRectangles(rects.tolist(), min(2, len(rects) - 1), eps=1)

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

                        # random_candidates = random_target_a(self._bboxes[bbox_ind], size=(n_candidates, 1)).astype('int')
                        # random_roi_feature = [self.extract_features(self.orig_col[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])], flag, inputShape, is_dl, is_olupdate) for r in random_candidates]
                        # random_roi_feature = np.array(random_roi_feature)
                        # pred_time = time.clock()
                        # print('predicting...')
                        # pred_random = self._model.predict(random_roi_feature)
                        # print('predict %s candidates took %s secs' % (n_candidates, round(time.clock() - pred_time)))
                        # thres = np.where(pred_random < 0.5)[0]
                        # if len(thres) > 0:
                        #     print('grouping candidates')
                        #     if len(thres) > 1:
                        #         rects = random_candidates[thres]
                        #         print(rects)
                        #         gp_rects, _ = cv2.groupRectangles(rects.tolist(), min(2, len(rects) - 1), eps=1)
                        #         print(gp_rects)
                        #         x, y, w, h = gp_rects[0]
                        #     else:
                        #         x, y, w, h = random_candidates[thres][0]
                        #         print(x, y, w, h)
                            
                        #     continue_prop = 1- pred_random[thres].mean()
                        # else:
                        #     print('no candidates has beetle')
                        #     gp_rects, _ = cv2.groupRectangles(random_candidates.tolist(), min(2, len(random_candidates) - 1), eps=1)
                        #     x, y, w, h = gp_rects[0]
                        #     continue_prop = 0

                        # cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.color[bbox_ind], 2)
                        # cv2.putText(self.frame, '# retargeting of %s: %s/%s' % (self.object_name[bbox_ind], n_try, N_MAX), (20, 35), self.font, 1, (0, 255, 255), 2)
                        # self._draw_bbox()
                        # cv2.imshow(self.window_name, self.frame)
                        # key_model = cv2.waitKey(1)
                        
                        # if key_model == 27:
                        #     print('break from the auto retargeting')
                        #     is_retarget = False
                        #     break

                    # section end for using group rectangle
                    print('Probability of beetle in bounding box %s: %s' % (self.object_name[bbox_ind], round(continue_prop, 3)))
                    n_try += 1
                    if n_try >= N_MAX:
                        # self._n = bbox_ind
                        # is_retarget = self._ask_retarget_box()
                        break
                    else:
                        is_retarget = False
                    self.frame = temp.copy()

                if not is_retarget:
                    # reset bounding box with highest probability
                    if n_try != N_MAX:
                        self._bboxes[bbox_ind] = (x, y, w, h)
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
    def _motion_detector(self, flag, inputShape, is_dl, is_olupdate, TEMP, N_MAX):

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
            if area > 200 and area < 800 and x < 1200 and (w/h < 2) and (h/w < 2):
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
            
            X_potential = np.array([self.extract_features(self.orig_col.copy()[y:(y+h), x:(x+w)], flag, inputShape, is_dl, is_olupdate) for x, y, w, h in potential_rect])
            
            if is_dl:
                pred_potential_prob = self._model.predict(X_potential)
                print('potential has no beetle probability: %s' % pred_potential_prob)
                pred_potential = np.array([1 if v[0] < 0.35 else 0 for v in pred_potential_prob])
            else:    
                pred_potential = self._model.predict(xgb.DMatrix(X_potential))
                pred_potential = np.array([1 if v < 0.5 else 0 for v in pred_potential])

            if any(pred_potential == 1):
                self._pot_rect = np.array(potential_rect)[np.where(pred_potential == 1)]
                self._pot_rect = self._pot_rect.tolist()
            else:
                self._pot_rect = []
        else:
            self._pot_rect =  []

class OnlineUpdateDetector(object):
    # generate postive image inside bounding box and update in record
    def _generate_positive(self, b, i, flag, inputShape, is_dl, is_olupdate):

        x, y, w, h = b
        img = self.orig_gray[y:(y+h), x:(x+w)].copy()

        if self.object_name[i] in self._record.keys():
            self._record[self.object_name[i]]['data']['positive'].append(self.extract_features(img, flag, inputShape, is_dl, is_olupdate))
            self._record[self.object_name[i]]['image'].append(img)
            self._record[self.object_name[i]]['num'] += 1
            self._record[self.object_name[i]]['loc'].append((x, y))
        else:
            self._record[self.object_name[i]] = {"data": {'positive': [], 'negative': [], 'rotate': []}, "model": None, "num": 1, "image": [img], "loc":[(x, y)]}
            self._record[self.object_name[i]]['data']['positive'].append(self.extract_features(img, flag, inputShape, is_dl, is_olupdate))
    
    # generate rotate positive data from current positive image and in record
    def _generate_rotate_pos(self, i, flag, inputShape, is_dl, is_olupdate):

        img = self._record[self.object_name[i]]['image'][-1].copy()
        n_ind = 1
        n_max = self._n_angle
        angle = 360 / n_max
        while n_ind < n_max:
            pos = rotate_image(img, angle, True)
            self._record[self.object_name[i]]['data']['rotate'].append(self.extract_features(pos, flag, inputShape, is_dl, is_olupdate))
            n_ind += 1
            angle += 360/n_max

    # generate negative data from surrounding bounding box with same size as positive and update in record
    def _generate_negative(self, b, i, flag, inputShape, is_dl, is_olupdate):

        x, y, w, h = b
        ratio = self._ratio
        neg_samples = [(max(0, int(x + w*rx*xsign)), max(0, int(y + w*ry*ysign)), w, h) for rx in ratio for ry in ratio for xsign in [-1, 1] for ysign in [-1, 1] if rx != 0 or ry != 0]
        for b_temp in neg_samples:
            x, y, w, h = b_temp
            img = self.orig_gray[y:(y+h), x:(x+w)]
            self._record[self.object_name[i]]['data']['negative'].append(self.extract_features(img, flag, inputShape, is_dl, is_olupdate))

    # train the i th bounding box model
    def _train(self, i):

        pos = self._record[self.object_name[i]]['data']['positive'] + self._record[self.object_name[i]]['data']['rotate']
        neg = self._record[self.object_name[i]]['data']['negative']
        X = np.array(pos + neg)
        Y = np.array([0] * len(pos) + [1] * len(neg))
        print('Samples size: %s  Positive: %s  Negative: %s' % (len(X), Y.sum(), len(X) - Y.sum()))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

        xg_train = xgb.DMatrix(X_train, label = Y_train)
        xg_test = xgb.DMatrix(X_test)

        model = xgb.train(params, xg_train, 100)
        
        y_pred = model.predict(xg_test)
        print("Detect model of %s's recall: %s  accuracy: %s" % (self.object_name[i], recall_score(Y_test, [round(v) for v in y_pred]), accuracy_score(Y_test, [round(v) for v in y_pred])))

        self._record[self.object_name[i]]['model'] = model

    # delete data generated from last record
    def _update_data(self, b, i, flag, inputShape, is_dl, is_olupdate):
        self._record[self.object_name[i]]['data']['positive'].pop(0)
        self._record[self.object_name[i]]['image'].pop(0)
        self._record[self.object_name[i]]['num'] -= 1
        del self._record[self.object_name[i]]['data']['rotate'][0:self._n_angle]
        del self._record[self.object_name[i]]['data']['negative'][0:(4*(len(self._ratio)**2) - 1)]

        img = self._generate_positive(b, i, flag, inputShape, is_dl, is_olupdate)
        self._generate_rotate_pos(i, flag, inputShape, is_dl, is_olupdate)
        self._generate_negative(b, i, flag, inputShape, is_dl, is_olupdate)

    def _detector(self, flag, inputShape, is_dl, is_olupdate):
        stop_obj = []
        # for all bounding box, collect data separately
        for i, b in enumerate(self._bboxes):
            if self._record.get(self.object_name[i], {}).get('num', 0) < 4:
                b = [int(elt) for elt in b]
                img = self._generate_positive(b, i, flag, inputShape, is_dl, is_olupdate)
                self._generate_rotate_pos(i, flag, inputShape, is_dl, is_olupdate)
                self._generate_negative(b, i, flag, inputShape, is_dl, is_olupdate)
                print('Collecting data for %s' % self.object_name[i])
                # return False, None
            else:
                b = [int(elt) for elt in b]
                x, y, w, h = b
                img = self.orig_gray[y:(y+h), x:(x+w)].copy()
                
                # ssim = compare_ssim(img, self._record[self.object_name[i]]['image'][-1])
                cos_sim = round(cosine_similarity(self.extract_features(img, flag, inputShape, is_dl, is_olupdate), 
                    self.extract_features(self._record[self.object_name[i]]['image'][-1], flag, inputShape, is_dl, is_olupdate))[0][0], 5)
                print('Similarity with last image of %s: %s' % (self.object_name[i], cos_sim))

                if cos_sim < 0.9999:

                    if not self._record[self.object_name[i]]['model']:
                        self._train(i)

                    X_current = np.array([self.extract_features(img, flag, inputShape, is_dl, is_olupdate)])
                    xg_current = xgb.DMatrix(X_current)
                    y_current = self._record[self.object_name[i]]['model'].predict(xg_current)

                    print('Probability of no beetle in %s: %s' % (self.object_name[i], y_current[0]))

                    # if the probability of no beetle in bounding box bigger than 0.5
                    if y_current[0] > 0.5:
                        n_try = 0
                        pred = np.array([1])

                        # start random retarget, if pred > 0.5, rerandom again
                        while pred[0] > 0.5:
                            if n_try == 0:
                                loc = self._record[self.object_name[i]]['loc']
                                nx, ny, nw, nh = loc[-1][0] + (loc[-1][0] - loc[-2][0]), loc[-1][1] + (loc[-1][1] - loc[-2][1]), w, h
                            else:
                                nx, ny, nw, nh = random_target2(b, 10)
                            
                            img = self.orig_gray[ny:(ny+nh), nx:(nx+nw)].copy()
                            X_current = np.array([self.extract_features(img, flag, inputShape, is_dl, is_olupdate)])

                            pred = self._record[self.object_name[i]]['model'].predict(xgb.DMatrix(X_current))
                            print('Probability of no beetle in new %s: %s' % (self.object_name[i], pred[0]))
                            n_try += 1
                            if n_try > N_MAX:
                                break

                        if n_try < N_MAX:
                            new_bbox = (nx, ny, nw, nh)
                            self._update_data(new_bbox, i)
                            self._train(i)
                            self._bboxes[i] = new_bbox # update new bounding box
                            print('Done retargeting %s...' % self.object_name[i])
                            
                            self._initialize_tracker()
                            self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]
                        else:
                            # self._ask_retarget_box()
                            del self._record[self.object_name[i]]
                            stop_obj.append(i)
                    
                # if image inside bounding box is similar
                else:
                    # update data every interval frame
                    if self.count % self._itv_f == 0:
                        if self._record.get(self.object_name[i], {}).get('num', 0) >= 4:
                            print('Update data and model of %s since this is frame %s' % (self.object_name[i], self.count))
                            self._update_data(b, i, flag, inputShape, is_dl, is_olupdate)
                            self._train(i)

        if len(stop_obj) != 0:
            return True, stop_obj
        else:
            return False, None

    # update model with incremental data
    def _update_model(self, type='stop'):
        if self._update:            
            x, y, w, h = self._bboxes[self._n]
            img = self.orig_gray[y:(y+h), x:(x+w)]
            if type == 'stop':
                xg_train = xgb.DMatrix(np.array([self.extract_features(img, flag, inputShape, is_dl, is_olupdate)]), np.array([1]))
            else:
                xg_train = xgb.DMatrix(np.array([self.extract_features(img, flag, inputShape, is_dl, is_olupdate)]), np.array([0]))

            self._model = xgb.train(params, xg_train, 50, xgb_model = self._model)
            print('Model updated')        

