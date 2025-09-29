The model operates using 2 libraries, dlib and yolov11, in accordance with opencv. What are those and how do they work?

Dlib is a machine learning library often used for real-time face detection and recognition. It works by analyzing images on a pixel-by-pixel basis, extracting facial landmarks (such as eyes, nose, and mouth positions) through analyzing light vectors in images. Any given image will most likely have darker and lighter sides to it. These differences in shading can be represented in thousands of vectors that point in the direction of more light. Certain facial features like eyes and ears have an easy to detect pattern of such vectors. Coupled with a database of public figures, this model can be trained to recognize the closest match to these patterns.s This system allows us to compare and recognize faces efficiently, even in live video streams.

YOLOv11 (You Only Look Once, version 11) is an object detection model. Unlike traditional approaches that scan an image multiple times, YOLO divides the image into a grid and predicts bounding boxes and class probabilities in a single forward pass. This makes it extremely fast and accurate for detecting multiple objects in real time.

OpenCV serves as the glue, providing tools to handle image input, preprocessing, and display — ensuring smooth integration between dlib and YOLOv11 in your model.
