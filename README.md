# opencv-tracker-interface

This is a multiple object tracker with human label interface created by OpenCV and Python. Given with proper trained model, it can auto stop while the tracker lost the objects.

## Version and Dependencies

* OS: Windows 7
* Python 3.5.2 
* [OpenCV 3.1.0](https://anaconda.org/menpo/opencv3)
* [mahotas 1.4.3](http://mahotas.readthedocs.io/en/latest/install.html)
* [scikit-image 0.13.0](http://scikit-image.org/docs/dev/install.html)

## Usage description
Type `python tracker.py` at command prompt and choose the video to initialze tracker by drawing target objects. The following are the current functions included while tracking:
1. Press 'r' to retarget
2. Press 'd' to delete bounding box
3. Press 'a' to add bounding box

Motion detection for adding bounding box automatically and better delete GUI are expected to be available in next version.

## Model
OpenCV build-in tracker has been already performing very well. However, since video of this project is grayscale and our target object is very similar with the context, human involvement to label target object is inevitable. Here is our procedures for training auto stop model:

1. Generate images that contain desired object by the tracker without auto stop model
2. Randomly generate  a bunch of image without object. 
3. Extract image descriptors like histogram, Haralick and Zernike Momemnts, etc
4. Train model (like xgboost) by labeling image without object as 1 and image contains object as 0

After adding model in the tracker, if model detects there is no object in the any bounding boxes then the tracker will stop tracking and let's user to decide the next move.


