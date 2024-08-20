# DL + Multiprocessing -> fast

'''
Issue 待解决: 
1. 只在初始frame[0]做了目标检测, 设定了相应的tracker, 但是之后的frame没有检测过 -> 如果有新的对象进入就检测不到
4. 多进程加速
5. 获取原视频播放的的FPS, 后按原FPS播放处理后的追踪视频
6. output video
'''

import cv2
from MOT_DL.utils import FPS
import numpy as np
import multiprocessing
from random import randint
import dlib

global trackerType_2
global trackTarget_2
global confidence_2
global videoPath_2
global output_2
trackerType_2 = ''
trackTarget_2 = []
confidence_2 = 0.3
videoPath_2 = ''
output_2 = False



def getValue(trackerType, trackTarget, confidence, videoPath, output):
    global trackerType_2
    global trackTarget_2
    global confidence_2
    global videoPath_2
    global output_2
    trackerType_2 = trackerType
    trackTarget_2 = trackTarget
    confidence_2 = confidence
    videoPath_2 = videoPath
    output_2 = output

def createClass():
    # Classes
    global CLASSES
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def createColor():
    # Colors
    global COLORS
    COLORS = []
    for i in np.arange(0, len(CLASSES)):
        while True:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            if color not in COLORS:
                COLORS.append(color)
                break

def multiProcessingDetect():
    return 0

def multiProcessingTracker(iq, oq, frame_rgb, label, bbox_coord):
    (startX, startY, endX, endY) = bbox_coord

    # Create tracker: dlib
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(startX, startY, endX, endY) # object initial postion
    tracker.start_track(frame_rgb, rect) # init tracker
    # trackers.append(tracker)

    while True:

        frame_rgb = iq.get()
        tracker.update(frame_rgb)
        pos_new = tracker.get_position()
        (startX, startY, endX, endY) = np.array([pos_new.left(), pos_new.top(), pos_new.right(), pos_new.bottom()]).astype("int")

        oq.put((label, (startX, startY, endX, endY)))


    # for (tracker, label) in zip(trackers, labels):
    #     # update tracker position
    #     tracker.update(frame_rgb)              
    #     pos_new = tracker.get_position()
    #     (startX, startY, endX, endY) = np.array([pos_new.left(), pos_new.top(), pos_new.right(), pos_new.bottom()]).astype("int")

    #     # Draw bbox
    #     idx = labels.index(label)
    #     cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
    #     cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[idx], 2)


    


def mot():   
    # in/output queues: trackers
    inputQueues = []
    outputQueues = []
    labels = []
    trackers = []
    frameCount = 0

    createClass()
    createColor()

    # Load model
    print("Info: loading model ...")
    prototxt = "MOT_DL/mobilenet_ssd/MobileNetSSD_deploy.prototxt" # model structure description file
    model = "MOT_DL/mobilenet_ssd/MobileNetSSD_deploy.caffemodel" # model weight file
    net = cv2.dnn.readNetFromCaffe(prototxt, model) # load trained model through Caffe Framework

    # Load video
    print("Info: starting video stream ...")
    cap = cv2.VideoCapture(videoPath_2) # define video source: video File / Camera

    # Write a new video <- output video
    # write = None

    # Create FPS timer (Frame Rate -> Frames Per Second) & start
    fps = FPS().start()

    #  Read and process video frames continuously
    while True:
        success, frame = cap.read() # read next frame from video stream
        if not success or frame is None:
            # capReadError = True
            break

        frameCount += 1

        # Resize
        frame_NN = frame
        (h, w) = frame_NN.shape[:2] # shape: tuple [height, width, channels]
        width_out = 1000
        fx = width_out / float(w)
        height_out = int(h * fx)
        dsize = (width_out, height_out) # formule: (width, height)
        frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA) # Area interpolation for image scaling
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB

        if len(inputQueues) == 0: # not detected, start detecting
            # Data Preprocess: frame pixel --transfer--> net input data
            '''
            Func: create a blob from frame -> blob: input layer of Net
                1. blob: Binary Large Object -> binary file which saves model weight or trained model 
                        -> 4 Dim Tensor [Num, Channels, Height, Width]
                        -> Num : Batch size -> num of samples in the blob
                        -> Channels: num of channels per sample
                2. scalefactor: 0.007843 = 1/127.5, 
                                compute: (Image pixel)/scalefactor, 
                                if 0, no scale
                3. size: expected scaled size
                4. mean: 127.5, average value -> image normalization, 
                        compute: (Image pixel) - mean, 
                        if omitted, not normalize 
                5. swapRB: if True, BGR. if False, BGR -> RGB
            '''
            blob = cv2.dnn.blobFromImage(frame_NN, 0.007843, (w, h), 127.5)
            
            # Train
            net.setInput(blob)
            '''
            Func: foward net
            detections: 4 Dim Tensor [A, B, C, D]
            -> A: Batch size -> num of input images processed at the same time
            -> B: Num of classes -> classes detected
            -> C: Num of bounding boxes per class -> bboxes detected per class detected
            -> D: (D0 - D3, D4, D5, D6) -> for each bbox detected 
                -> D0-D3: bbox coordinates: 
                            X_left_up, Y_left_up, X_right_down, Y_right_down
                    or    X_left_up, Y_left_up, box_width, box_height
                -> D4: confidence score
                -> D5: class probability
                -> D6: class index
            '''
            detections = net.forward()
            # batch_size = detections[0]
            # num_class = detections[1]
            # num_bbox_per_class = detections.shape[2]
            # info_per_bbox = detections[3]

            # Result processing
            for ibbox in np.arange(0, detections.shape[2]): # for each bbox detected in one class
                confidence = detections[0, 0, ibbox, 2] # here 2 is confidence

                if confidence > confidence_2:
                    idx = int(detections[0, 0, ibbox, 1]) # here 1 is class index
                    label = CLASSES[idx]

                    if label not in trackTarget_2:
                        continue

                    labels.append(label)                

                    (startX, startY, endX, endY) = (detections[0, 0, ibbox, 3:7] * np.array([w * fx, h * fx, w * fx, h * fx])).astype("int") # bbox coordinates (nomorlized -> ratio): D3,D4,D5,D6 (D7: no)
                    bbox_coord = (startX, startY, endX, endY)

                    # Object Tracker: dlib
                    # tracker = dlib.correlation_tracker()
                    # rect = dlib.rectangle(startX, startY, endX, endY) # object initial postion
                    # tracker.start_track(frame_rgb, rect) # init tracker
                    # trackers.append(tracker)

                    # Multi Processing
                    iq = multiprocessing.Queue() # input queue
                    oq = multiprocessing.Queue() # output queue
                    inputQueues.append(iq)
                    outputQueues.append(oq)

                    # execute process
                    process = multiprocessing.Process(
                        target=multiProcessingTracker, name="multiProcessingTracker", 
                        args=(iq, oq, frame_rgb, label, bbox_coord), 
                        daemon=True)
                    process.start()

                    # Draw bbox
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[idx], 2)

        else: # detected yet, start tracking
            
            for iq in inputQueues:
                iq.put(frame_rgb)
            
            for oq in outputQueues:
                # update tracker position
                (label, (startX, startY, endX, endY)) = oq.get()

                # Draw bbox
                idx = labels.index(label)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[idx], 2)


            # for (tracker, label) in zip(trackers, labels):
            #     # update tracker position
            #     tracker.update(frame_rgb)              
            #     pos_new = tracker.get_position()
            #     (startX, startY, endX, endY) = np.array([pos_new.left(), pos_new.top(), pos_new.right(), pos_new.bottom()]).astype("int")

            #     # Draw bbox
            #     idx = labels.index(label)
            #     cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            #     cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[idx], 2)

        # Show frame
        cv2.imshow("MultiTracker", frame)
        if cv2.waitKey(1) & 0XFF == 27: # 'Esc' -> quit on ESC button
            break
        elif cv2.getWindowProperty("MultiTracker", cv2.WND_PROP_VISIBLE) < 1:
            break

        fps.update()

    fps.stop()

    print(f"Labels: {labels}")
    print("[INFO] elapsed time: {:.2f}s".format(fps.elapsed()))
    print("[INFO] Average FPS: {:.2f} frames/s".format(fps.fps()))

    cv2.destroyAllWindows()
    cap.release()
