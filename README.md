# opencv-tracker-interface

This is a multiple object tracker with human label interface created by OpenCV and Python. Given a proper trained model (like a keras model in model directory in this repository), it can auto stop while the tracker lost the objects and auto retarget the objects.

## Version and Dependencies

* OS: Windows 7
* Python 3.5.2 
* Keras 2.0.4
* [OpenCV 3.1.0](https://anaconda.org/menpo/opencv3)

## Usage description
Type `python main.py` at command prompt and choose the video to initialze tracker by drawing target objects. 
Press 'h' in this tracker for more details.

## Model
OpenCV build-in tracker has been already performing very well. However, since video of this project is grayscale and our target object is very similar with the context, human involvement to label target object is inevitable. Here is our procedures for training auto stop model:

1. Generate images that contain desired object by the tracker without auto stop model
2. Randomly generate  a bunch of image without object. 
3. Extract image descriptors like histogram, Haralick and Zernike Momemnts, etc
4. Train model (like neural network or xgboost) by labeling image without object as 1 and image contains object as 0

After adding model in the tracker, if model detects there is no object in the any bounding boxes then the tracker will random a number of some candidates. If model predicts that there is beetle inside the candidate bounding box,  tracking process continue with the candidate bounding box. Otherwise, tracking will stop and let's user to decide the next move.

In our case, we use and transfer ResNet50 as our predictive model.

## TO-DO
The precision and recall of our autoupdate tracker is about 98% but the speed is around 1ps, I am trying the another state-of-art object detector [YOLO](https://github.com/philipperemy/yolo-9000) and modified it to suit our case for faster performance.

