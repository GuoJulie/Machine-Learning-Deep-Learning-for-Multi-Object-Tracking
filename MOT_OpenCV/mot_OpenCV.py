# OpenCV - multi object tracking
# algrotihm: OpenCV -> 'Boosting','CSRT (recommended)','KCF (recommended)','MedianFlow','MIL','MOSSE','TLD','GOTURN'

'''
RUN Steps:
1. SELECT:    press "SPACE" to select ROI
2. PLAY:      then press "SPACE" or "ENTER" button to continue to play video
3. EXIT:      press "ESC" or close windows
'''

'''
ISSUE:
1. 当前: 视屏暂停后, 可以直接鼠标画框选定selectROI, 选定好之后视频继续播放, 或者待实现: 也可以直接按space空格键, 视频就继续播放
2. 初始状态就是暂停 -> 可选ROI or SPACE继续播放
3. output video
'''

from random import randint
import cv2

global trackerType_1
global videoPath_1
global output_1
trackerType_1 = ''
videoPath_1 = ''
output_1 = False

def getValue(trackerType, videoPath, output):
    global trackerType_1
    global videoPath_1
    global output_1
    trackerType_1 = trackerType
    videoPath_1 = videoPath
    output_1 = output

# Tracker selection logic
TRACKER_TYPES = {
    'Boosting': cv2.legacy.TrackerBoosting.create,
    'CSRT (recommended)': cv2.legacy.TrackerCSRT.create,
    'KCF (recommended)': cv2.legacy.TrackerKCF.create,
    'MedianFlow': cv2.legacy.TrackerMedianFlow.create,
    'MIL': cv2.legacy.TrackerMIL.create,
    'MOSSE': cv2.legacy.TrackerMOSSE.create,
    'TLD': cv2.legacy.TrackerTLD.create,
    'GOTURN': cv2.TrackerGOTURN.create,
}

# Run: tracking
def mot():
    multiTracker = cv2.legacy.MultiTracker.create()    
    cap = cv2.VideoCapture(videoPath_1)
    # Select boxes list
    # bboxes = []
    colors = []
    
    # Run: track objects
    while True:
        # Read video
        success, frame = cap.read()
        if not success or frame is None:
            # capReadError = True
            break

        # get updated location of bboxes in subsequent frames
        success, bboxes = multiTracker.update(frame)
        # print(f"Updated bounding boxes: {bboxes}")

        # Draw updated boxes
        for i, newbox in enumerate(bboxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # Show frame
        cv2.imshow("MultiTracker", frame)

        if cv2.waitKey(1) & 0XFF == 32: # 'SPACE' -> select ROI
            print("Please draw a rectangle box on the object you want to track. Then press 'SPACE' or 'ENTER' button")
            
            bbox = cv2.selectROI("MultiTracker", frame, fromCenter=False) # tuple
            print(f"selected ROI: {bbox}")
            # bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

            tracker = TRACKER_TYPES[trackerType_1]()
            multiTracker.add(tracker, frame, bbox)
            
        if cv2.waitKey(1) & 0XFF == 27: # 'Esc' -> quit on ESC button
            break
        elif cv2.getWindowProperty("MultiTracker", cv2.WND_PROP_VISIBLE) < 1:
            break
