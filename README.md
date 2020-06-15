# Object-Tracking-using-YOLOv3

Detecting Objects on Image with OpenCV deep learning library

Algorithm:
Reading RGB image --> Getting Blob --> Loading YOLO v3 Network -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels

Result:
Window with Detected Objects, Bounding Boxes and Labels

"""
Some comments

With OpenCV function 'cv2.dnn.blobFromImage' we get 4-dimensional
so called 'blob' from input image after mean subtraction,
normalizing, and RB channels swapping. Resulted shape has:
 - number of images
 - number of channels
 - width
 - height
E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
"""

