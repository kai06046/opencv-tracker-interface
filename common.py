import numpy as np
import os, sys
from skimage.feature import local_binary_pattern
# some common function
# convert (x0, y0, width, height) into ((x0, y0), (x1, y1)), where x1 = x0 + width; y1 = y0 + height
convert = lambda x, y, w, h: [(int(x), int(y)), (int(x + w), int(y + h))]
# add randomness to an integer
vary = lambda x, var, flag: x + abs(np.random.randint(-var, var)) if flag else x + np.random.randint(-var, var)
# random a new bounding box with a bounding box as input
def random_target(bbox, var = 30, r = 0.3, flag=False):
    x, y, w, h = bbox
    # x1, y1 = vary(x, var), vary(y, var)
    x1, y1, w1, h1 = vary(x, var, flag), vary(y, var, flag), vary(w, int(var*r), flag), vary(h, int(var*r), flag)
    return int(max(0, x1)), int(max(0, y1)), int(max(1, w1)), int(max(1, h1))
# get specific line from a text file
def getlines(txt, n_line):
    with open(txt, 'r') as f:
        lines = f.readlines()
    try: 
        line = lines[n_line]
    except:
        line = '[%s, 0, []]' % int(n_line + 1)
    return line
# see if a points is inside a rectangle
def in_rect(pt, rect):  
    x_condition = pt[0] > rect[0][0] and pt[0] < rect[1][0]
    y_condition = pt[1] > rect[0][1] and pt[1] < rect[1][1]
    
    if x_condition and y_condition:
        return True
    else:
        return False
# see if two rectangle was overlapped
def rect_overlap(rect1, rect2):
    rect1 = np.array(rect1).transpose().tolist()
    rect2 = np.array(rect2).transpose().tolist()
    
    return range_overlap(rect1[0], rect2[0]) and range_overlap(rect1[1], rect2[1])
# see if range was overlapped
def range_overlap(x1, x2):
    return (x1[0] <= x2[1]) and (x2[0] <= x1[1])
# see if the rectangle is overlapped
def overlapped(rect, rect_list):
    return any([rect_overlap(rect, r) for r in rect_list])
# calculate area of a rectangle
def area(r):
    x = abs(r.p1[0] - r.p2[0])
    y = abs(r.p1[1] - r.p2[1])

    return x*y
# calculate intersect area of two rectangles
def intersect_area(r1, r2):
    
    dx = min(r1.p2[0], r2.p2[0]) - max(r1.p1[0], r2.p1[0])
    dy = min(r1.p2[1], r2.p2[1]) - max(r1.p1[1], r2.p1[1])

    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return 0

def dir_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Done creating %s' % path)
    else:
        print('%s exists' % path)

def find_data_file(filename):
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)
 
    return os.path.join(datadir, filename)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# local binary pattern histogram feature
class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist
lbs = LocalBinaryPatterns(12, 4)

# center a tkinter widget
def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

def on_closing():
    if askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
            