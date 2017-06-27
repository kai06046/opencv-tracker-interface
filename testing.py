import cv2
import numpy as np
import time
from multiprocessing.dummy import Pool as ThreadPool

# from keras.models import load_model
# model = load_model('model/nadam_resnet_first_3_freeze_3.h5')
pool = ThreadPool(4)

def extract_features(img, inputShape):
    img = cv2.resize(img, inputShape, interpolation=cv2.INTER_NEAREST)
    img = img / 255.0

    return img

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print('y2: %s' % y2)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    # print('before sort ids: %s' % idxs)
    idxs = np.argsort(idxs)
    # print('ids: %s' % idxs)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        # print('i: %s' % i)
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # print('pick: %s ' % pick)
    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def vary_2(x, var, flag, size):
    if not flag:
        return x + np.random.randint(-var, var, size)
    else:
        return x + np.random.randint(0, var, size)

# random a new bounding box with a bounding box as input    
def random_target_a(bbox, var = 35, r = 0.4, flag=False, size=(10, 1)):
    x, y, w, h = bbox
    # x1, y1 = vary(x, var, flag), vary(y, var, flag)
    x1, y1, w1, h1 = vary_2(x, var, flag, size), vary_2(y, var, flag, size), vary_2(w, int(var*r), flag, size), vary_2(h, int(var*r), flag, size)
    
    random_candidate = np.hstack((x1, y1, w1, h1))
    random_candidate[random_candidate < 0] = 1
    
    return random_candidate

video = cv2.VideoCapture('videos/[CH04] 2016-10-20 19.20.00_x264.avi')

# video = cv2.VideoCapture('videos/[CH04] 2016-09-28 20.20.00_x264.avi')
# background subtractor
bs = cv2.createBackgroundSubtractorMOG2()

while True:

    ok, frame = video.read()
    
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(1)
    
    potential_rect = []
    expand_rect = []

    fg_mask = bs.apply(gray.copy())

    th = cv2.threshold(fg_mask.copy(), 200, 255, cv2.THRESH_BINARY)[1]

    # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)), iterations=5)
    # get contours from dilated frame
    _, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        

        area = cv2.contourArea(c)
        # area = w * h
        
        if area > 200 and area < 800 and (w*h < 3600) and (w / float(h) < 1.8) and (h / float(w) < 1.8):
            potential_rect.append([x, y, w, h])
            if area < 350:
                x = int(x / 1.02)
                y = int(y / 1.02)
                w = int(w * 2.2)
                h = int(h * 1.6)
                expand_rect.append([x, y, w, h])
            # if area < 400:
            #   random_candidates = random_target_a((x, y, w, h), flag=True, size=(10, 1)).astype('int')
            #   # random_roi_feature = [np.expand_dims(extract_features(frame[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])], (224, 224)), 0) for r in random_candidates]
            #   random_roi_feature = [extract_features(frame[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])], (224, 224)) for r in random_candidates]
            #   pred_time = time.clock()
            #   print('predicting')
            #   pred_random = model.predict(np.array(random_roi_feature))
            #   # pred_random = pool.map(model.predict, random_roi_feature)
            #   # pred_random = np.vstack(tuple(pred_random))
            #   print('predict took %s secs' % (round(time.clock() - pred_time)))
            #   thres = np.where(pred_random < 0.5)[0]
            #   gp_rects, _ = cv2.groupRectangles(random_candidates[thres].tolist(), 2, eps=1)
            #   if len(gp_rects) > 0:
            #       x, y, w, h = gp_rects[0]
                # potential_rect.append([x, y, x+w, y+h])
            

    # print('pr %s' % potential_rect)
    # gp_rects, _ = cv2.groupRectangles(potential_rect, max(4, len(potential_rect) - 2), eps=100)
    # gp_rects = non_max_suppression(np.array(potential_rect), overlapThresh=0.1)
    # print('gp %s' % gp_rects)
    # for b in potential_rect:
    #     x,y,w,h = b
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
    for b in expand_rect:
        x,y,w,h = b
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # for b in gp_rects:
    #   x,y,w,h = b
    #   cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    if key in [ord('p'), ord(' ')]:

        while True:
            key2 = cv2.waitKey(1)
            cv2.imshow('frame', frame)
            if key2 in [ord('p'), ord(' '), 27]:
                break

    cv2.imshow('frame', frame)
    cv2.imshow('di', fg_mask)
    if key == 27: break

video.release()
cv2.destroyAllWindows()

