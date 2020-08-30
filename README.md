# Object-Tracking-using-YOLOv3


<img
src = "https://i2.wp.com/kobiso.github.io/assets/images/yolo/yolo%20model.png?w=750&ssl=1" width = "500" height ="380"/> <img src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMoAAAD5CAMAAABRVVqZAAAA51BMVEX///8B/wEBAf//AQEFBQUAAACPj4/6+vrV1dVHR0fJycnf39/p/+nCwsLu7u65ublMTEzo6Oienp6rq6s2Njb19f+Ghob/9/f/zs7/5eV+fn6zs7P2//aYmJhXV1dsbGz/kZEv/y9a/1qo/6jS/9K1tf8TExP/v7//t7fe3v9KSv//e3vH/8dpaf//XV1p/2nq6v9cXP//h4ef/59A/0De/968/7zJyf8dHR3/29v/LS2dnf9O/05+/36Kiv+UlP//cHD/pqZCQv8xMf+mpv//ICD/OjqAgP/U1P8hIf//S0uP/49zc/8Zm+mZAAAP5klEQVR4nO1de18aORdG1mG4CMhF2wUBUYu1N7e1tVprq9Ru27Xf//O8zCSZJJPknCQMM/T38vxVmMlwnuTck7Gl0grx5Pr0/vDdy7u7u5fv7p8//XuVv7VCXN/f/aXg08vnfxqfUw0Nhu+H10WLZ43rl2YedHHunxQtpA1O/8WIxLhbe0U7/WRFZP3JPLVbEYZ3a6tmTwBbN+B50TLr8dSZyAJv1nFh3vkwWeBp0YKn8cTNSkQcFi27jGtvIgu8KVp6EV5mwvFf0fJzPF+OySL6r4vxny7LZMElLJpEjCW1i+DfollEWMriOdbA9p9kw2SRxRTNpOQfT9IoOonxjfE6FJspZ2LyDMWafpZMik1h0MrXEcWpWEZ+mKM4j/xf1lT+Oi2ISaY2T/CpICrZhRSOYpZlBYtSlEN+Yyfc9zfvDg8PX9raVRH18d9Wk3zP/ev1oc2IIpyYhWBv0s3h59/xQQVUYahM33W6gk/Afe5MUKM3aAoaVvM3fCwlNpYfTzAlyz17QdrcL80jsWItbw1D5AH9EOL78vZhSJcFdkNIsyknCgywJ8LSDzi65ry/BwqDOiHYjeVc5IOy4MkHOBP59l5Aq7fI1MGolK/dgxpyaPEAaPz3VUsvAXRgNskt2BVYufgi7peVBPTHuWaUEBWrohbU0FxTFyisWFkt6DdypQIlk3dWT9hQyR5LK1j4R1Cx2iRdH7Nf2hmDgSlXZwyGBZtJBTPrlYsvAsyhbMpAsLO5cvFFgIWghbEsOz5LQKJYaBioX0BfYBUAuxS4LOBM5NyngPe7sGWBy+mc28agN8aiJNJyybnVijQZYR2BN2Zyb0/CVMCeC3LE8jAvCpbyAAqPbSvnvsOCnpwytYDQHaZcacTAJNK75Gt0hyXnqBIBP1SssV+LrbICdvDwXVWd6aPqlW/niAJTFW0qhS5L/pteJSxKmrrY2A5eMYcoYZlM5guPOsyTAAe8LKb0A95cKeqULiSTWeehtOUwP+FlABMMtCih9C0/2dMwTzAUHcwRqcCToMYJBrN8Y5O10LPGJtcK116mUcWezNerGLYFp4+uBR801kdvLM5p0+q7POR1lQpPPjSnw4o6ayhAVXyLilbjL9bhFRalKvTai1yP94pTceLOZkzaIa/Le3hyEWKnKfdryURel0PLMZ/Wkom0oWc7RKhC18NOGBJ9sY9zTC3X5vU7BupeHVqLfzt4iZzxxlVXYq0s6oUCGKeuU7ymL3bHuHOT7HnRb6ptsMEGFjjYOShahGWx8+ri/P0Ww/To9vjkT+S0c3y+pcP09lXRojnhw+NUy4Pi/LJoAW1x+R7iQXC7U7SUFjjGecQ4OilaUgS2RGIyH4qWFsDJ1IHJAv7+7OzG7f4fe063H+h9lhkXbuII+FIu/+Ny/7Ny+ZnD7ZeORLa2HOVPcParXC6/dRlRjgZYb/PdOjPxdck35Rgf7Ud8ISN+Wt18YOGAU/jsyeQ1kav8zXrEP3RE+cHi5h1nIltbfv5r71vZRbAYb5Mhv9F7P3gwufVikkxwBEuv9FEYglmYDxM/R3wmMin/sBv0q2zNxYuJlyOW1mSBM5tBD/IYiIuPnWxNfZjs/Srbi5UMSo0pvzbfO/Wh4pXpf0tLZeOQfyuDvphu/ezD5MiHyWtFqPIvdNCZOsgU+B99mGz5pPg3GqHKaCr2QjdKa2OvvJj4OGLd9JZRh/xRO0i3mAdeTLySr7TJUyBBTz9I58Zdc2GCYw8mXwxCwQ75q2mUko6deDHxccQG9VpgDoxSHHECJYGbelHxqYTnRqGgQkTj8xhS/sKl+uU492Cit14K46if0CjZX3gx8XLEBpsn+GoaBSxlKlD6LcqjB5NnkEzG2hgZJS6L36J4MNFkLBIMWRU8SFxMy1r+6PHyZIHLW1Jn+lTBoKVE0Fa6RkfMwG+1KYGPxLTx4Hi69d6DiVAFGqCrjdMFgYrE9Vnk9ueKhV/6VMHm6JBAkyGrGXEaSYlwgTLJqmX/gAqlzfVRBUvcxRQh8j6zBvcckWhuSClRE6N9Dky/vCoSLTD9emEcac52CKiGIUHFy7z1QMIDlINhXMhyIjlxhht1iP2CBQuiY8TGYCZZ7p2ASQtW3cPzEEdJ2FR8MkYTYFMB2icxQnB0bGZwqM9ydw7WEXRrAvbk0R1gVMlyUWBZsEXBliWaCdDqM91lBLXdog/2AxsPdr+yZAInYBbjQQWN0rDc9AtM8HH9KsG5/gNCxaeh4i8JijnwgB8IlWw35CEqVnt4kLH9Rlp5mW7Hgx7IalMCypBfI1SsosrBDgDhPjBCWlHR9pop3mZBBWqbi35jtVQiv7G0gkF745ILhKhY2QqiYCAVq+rxCHiA1OiHqFidkABalCgVq21G6AFSqwyiYmzmiYACU+SMp4AkNgUkmFlLgWkOSGIuIDlAY4sCE6QfNokLmFlLvTIwB7PYuQeL0EhDwRMtFm07MIeT/IZxY4WJggAylbgXCJb2+OkVuHKTqmlwVqHCngCu3KJVhfcg0dQF7gxIt8KtBtQdgzl+nFnDm5BYvwXeLZMza7h2wo4fwRNBukegMFhyPHUZPAelQXa7tXvdqcHISTYw4ruNBe0eOa+GNFuJemJbEkAihnWbU7dj/VKgU4F0A2kRiu7YG7lgTNJFKGwsEBeMCWvlo2dbDG4MPWOpRCV0e8XgxtBWPotK+KaXbtfxwxQd5jy5+rNE/8AWHyHJFVCZ1AnesTj2qmlyoDKpYX8PcRYR+Gkyq0Mhj4I7ugQTNwaNXuIbWAt8Feqwj2C2wsD10vZUyNHtxfHxxe3U7m5ddMX2FpJp/vLw8HDze253t3j8yP1osQW0qSiu9h4QO09+Z8EQ6JjgW3E+kH5hBctiqA/m2TORMx6/M1QQTOerwIM3fkj9gpVLcoGxw2HlklyQ9t5eR4wBmBvn+NkIN6hdAfwcghOAzVj8HIITNHlbppYP9gQydci6LYAsVQzel8lSxfRNJ/fXokzATodmGFwMPSe/498aoF1ziwzRDsbCMyOPbNFpRgsXOwDNs0xM3+r03jwLJmCjebo8E7s3c0LkEKUNjK98xDhYmovtO0bhfLVMSkvrmMOe8pJcLLYxlrJ9p5OJS2VjVntLS/hkx8M9+JlII6z2Lv0LMfeTib6x8oX1290HXkrmc/Rizysfs9rtY3BPYo48DybirbE05pbKlcDx9W7/k8ghvG+iwOVte4odh1emljvWc+aQxjjpFscHSzLLn086s2r1lcs3/n8afAd3zEfZHHLfwx3zCw/VknAC/U2az5cZnkL+CRnN/MHpL1iYsHN5rqHz+fFV5n/26OxGZzbffj9z+yshMA5OLi9ujz7HOH+8eLW6P6mz9/Ph6++3LyK8ffvj5pmr691ggw022GCDDf6P0IpQqARhJMEy/9NPqzma1GYBw9XuoNfITDorhPXuZPcqkSCoDYZ0TsPaboTaSDdsn15kn4c1On6bgHyYjfL7b5CGu7IERITdanxxTK/pBjLi5FN1xp8gYnFHMx8idaME3ZgnuRhU1ZFNeqlLnqN/DHlULlyagAT1xfUWlbeiDp3QS534k/k5EZccdKwFSBBcRXfU4huCmTqWDCV3lXqygQSyzupmImtMtBKIy0JlDBRPVKcXiEeo0U/B1aA7bFYbjUa93Rv06bN0M5ExwoBL0CMCNHv7V5RLsL+4pWPSsIHIMaQfrlI21embZiJrMKPud6SvR+zr6ENf1CMBgah5dcluBNDJCnqrIcBB9Eu1yl36ffTvkX5e69JqdQPOXfuowSrEF0FUKZikv2erFVFs6DWM6RfRqIlR4Aq5UlOvZAq2+kosr1I545h/pdWwmbBwiZ/rqr8xNHrARrPdruoytbBTrTeb9WrH7MI79XZTGstMWolgHVH1KzoNY2T3yUc6J5pQWBd1uJKkDnWWRgQT2VN0Rn3uSvsV4UfJV4u7OwN2fZBcbkhKIlEhaIhSS2u3Lw8NuPc2zEqLU7kqhbuJy1/8Y8xnPxynM6hJcpFRGYlj96UZoz8jIqwTkMfMNBp2JelXyzQnnEqDUCHOYSYFZu4tNBGb/yz5xc5IuofZOcuikLRioGoYW8+BLK/iiznLOqcyHgfbcixmc8uikBiqE19CqES5njg4aMcX23ZU6qqGVWSNarhQiSXZ3p1MxkkyQCVgPxT0x/xqEiqoK4yT8cmEqShdNUsqzKiFqCHrV2JOGiosD2iKM8D8Q4sFrWH8kbr0CpUn7ErhNUlM6OAeFavjQmWSljSlX8l06h4kOjfVGXbFyBbfKka5gfgFS0GSHyHpIZkGS1tJ7ks0bCRqDUwlVKmI8ZJcJlpCLE70HfFzg22JirDyZGzXhYqiYX1Zv1ypiG49Dq7EHmgIEvxpi19ktjIWxsZqTtIQSC8kpDSMOaxEFSyoiGbfFq7vc/mpwkiDhZ9VE4oap1K1pcKMij6nm84SACo6DybmDSP+FbWbRjVBXaUyFMbuciodWyrMDVENY/qVjAPmRIqeFVE0giE3EJZBpItAaj3qivpQKY1FDVP0K/FomuRQjfYylTbXv30l1G8LKwpTaYkCgmBZf+ziWY3MnyrJq6cSwlSaJe2qEDRxKswkVQl63RiJMTAN202eIBVskkHIqIo326xKsFtLoW9BpSR6Fwk00ealVI1PLpsA0S2ak3wqKqlXzLYSiTBSPJgEmEpf9QoEY5pCJF8wpWryaCSW69uqp6QYiSFJ48G6/Cv6T5PlwlQmaYkZ+ukLTIUGPFEWbXwi5h+6SZmIVMTFq/C4MlTXzJ7KvqoqdJgyy7zxMlOyD57yagQQn1RRF2+iRHshcQl7MRoWVHoGCVicEMaxsNhhLklqBzX5oslgS1gXqewKN8SqKeVgwoNpNdu2oJKq0BMwIxcmiIWOXsJJHJD0Bmu9RqLrYWeY1FKhQEU0lqaofyQ9FGqJpmhbMBUuQZtL1mqnBCCgJcq4JtpxAtaxJd2F2rjWv5oJpR41IkZlxrhUpY4ftatkacN+IIgBU2FRPJZg1h+P+31RAI05bDPyKWdV5ZFa6T1zd588I6jtD9vDSo0VisSFMDcZDLrt9nA0CSR9RKi0NRLwbyRxBWEV/RKWRYfEs1Fb6bPynM0Z8ztJgSylYHUrKqUrswTp/qvY6dC0VGum/Y3FErAHsbiyLU9LMi+a3Z4gERWj0gnMEqRi90Ckotkw6Qbqs+K55S6JBZGW3Ozi3mWYesTiYxK/MSqlcKKRIHrELJ3O1EUq2k2G9kTYmyXoDzTxsFMKK1yLBuLih92+NFzYkyXfiJkJuVVwv63eeBakcDXQpFPiDTomsSjVZm9U2V+g0m3XG6kkRMzBqsPFbZWe2gUM6+34CZVeXSoamnGXUfyqGn+TmtVWvd2tUAmGzao+DWrUE3ju/GjSyT8VGyrriA2VdcSGyjpiQ2UdsaGSH/4HUfWJtWzSXbIAAAAASUVORK5CYII="
### Detecting Objects on Image with OpenCV deep learning library

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

### Detecting Objects on Video with OpenCV deep learning library

Algorithm:
Reading input video --> Loading YOLO v3 Network -->
--> Reading frames in the loop --> Getting blob from the frame -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
--> Writing processed frames

Result:
New video file with Detected Objects, Bounding Boxes and Labels

"""
Some comments

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
"""

### Detecting Objects in Real Time with OpenCV deep learning library

Algorithm:
Reading stream video from camera --> Loading YOLO v3 Network -->
--> Reading frames in the loop --> Getting blob from the frame -->
--> Implementing Forward Pass --> Getting Bounding Boxes -->
--> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
--> Showing processed frames in OpenCV Window

Result:
Window with Detected Objects, Bounding Boxes and Labels in Real Time


"""
Some comments

cv2.VideoCapture(0)

To capture video, it is needed to create VideoCapture object.
Its argument can be camera's index or name of video file.
Camera index is usually 0 for built-in one.
Try to select other cameras by passing 1, 2, 3, etc.
"""
