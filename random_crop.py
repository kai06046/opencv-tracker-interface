import numpy as np

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def random_crop(img, area_ratio, hw_vari):
    """ 
    random cropping
    ===============
    area_ratio: the ratio of cropped image
    hw_vari: variance of scale ratio (0 ~ infinite)
    
    """

    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta

    
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
    h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
    
    # cropped width and height can't bigger than original width and height
    if w_crop > w:
        w_crop = w
    if h_crop > h:
        h_crop = h

    # random generate original of cropped image
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)

    return crop_image(img, x0, y0, w_crop, h_crop)