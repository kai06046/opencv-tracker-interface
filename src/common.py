import numpy as np
import os, sys
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askokcancel
# some common function
# convert (x0, y0, width, height) into ((x0, y0), (x1, y1)), where x1 = x0 + width; y1 = y0 + height
convert = lambda x, y, w, h: [(int(x), int(y)), (int(x + w), int(y + h))]

# add randomness to an integer
def vary(x, var, flag, size):
    if not flag:
        return x + np.random.randint(-var, var, size)
    else:
        return x + abs(np.random.randint(-var, var, size))
# random a new bounding box with a bounding box as input    
def random_target(bbox, var = 35, r = 0.3, flag=False, size=(10, 1)):
    x, y, w, h = bbox
    # x1, y1 = vary(x, var, flag), vary(y, var, flag)
    x1, y1, w1, h1 = vary(x, var, flag, size), vary(y, var, flag, size), vary(w, int(var*r), flag, size), vary(h, int(var*r), flag, size)
    
    random_candidate = np.hstack((x1, y1, w1, h1))
    random_candidate[random_candidate < 0] = 1
    
    return random_candidate

# get specific line from a text file
def getlines(txt, n_line):
    with open(txt, 'r') as f:
        lines = f.readlines()
    try: 
        line = lines[n_line]
    except:
        line = '[%s, 0, [], [], []]' % int(n_line + 1)
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

def get_path():
    root = tk.Tk()
    root.withdraw()
    path = askopenfilename(title='Where is the video?', filetypes=[('video file (*.avi;*.mp4)','*.avi;*.mp4')])
    if path == "":
        if askokcancel('Quit', 'Do you want to quit the program?'):
            root.destroy()
            sys.exit()
        else:
            root.destroy()
            return get_path()
    else:
        root.destroy()
    root.mainloop()

    return path

#####################################################

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    
    # 旋转角度的周期是360°
    angle %= 360
    
    # 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

    # 如果需要裁剪去除黑边
    if crop:
        # 对于裁剪角度的等效周期是180°
        angle_crop = angle % 180
        
        # 并且关于90°对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
            
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180.0
        
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        
        # 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        
        # 计算分母项
        denominator = r * tan_theta + 1
        
        # 计算最终的边长系数
        crop_mult = numerator / denominator
        
        # 得到裁剪区域
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)