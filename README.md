# Object Tracking using YOLOv3


<img
src = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRBHC8oSbf3Jq0MDoLV1PVfhrFVVPN-ptjAdQ&usqp=CAU" /> 

## Refer my blog for detail explanation
## 👉 [YOLO : Real Time Object Detection](https://capablemachine.com/2020/07/21/yolo-model/)
### Detecting Objects on Image with OpenCV deep learning library

Algorithm:
Reading RGB image --> Getting Blob --> Loading YOLO v3 Network -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels

Result:
Window with Detected Objects, Bounding Boxes and Labels.


#### Some comments

With OpenCV function 'cv2.dnn.blobFromImage' we get 4-dimensional
so called 'blob' from input image after mean subtraction,
normalizing, and RB channels swapping. Resulted shape has:
 - number of images
 - number of channels
 - width
 - height
E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)


### Detecting Objects on Video with OpenCV deep learning library

Algorithm:
Reading input video --> Loading YOLO v3 Network -->
--> Reading frames in the loop --> Getting blob from the frame -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
--> Writing processed frames

Result:
New video file with Detected Objects, Bounding Boxes and Labels.


#### Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.


### Detecting Objects in Real Time with OpenCV deep learning library

Algorithm:
Reading stream video from camera --> Loading YOLO v3 Network -->
--> Reading frames in the loop --> Getting blob from the frame -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
--> Showing processed frames in OpenCV Window

Result:
Window with Detected Objects, Bounding Boxes and Labels in Real Time.


#### Some comments

cv2.VideoCapture(0)

To capture video, it is needed to create VideoCapture object.
Its argument can be camera's index or name of video file.
Camera index is usually 0 for built-in one.
Try to select other cameras by passing 1, 2, 3, etc.

