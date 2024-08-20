import os
import tkinter as tk
from tkinter import filedialog, messagebox

'''
Improvements:
1. cap 视频流 可设为 摄像头实时输入 (待改进  OpenCV + DL)
'''

# Global variables 
global trackerType
global videoPath
global trackTarget
global confidence
global output
trackerType = ''
videoPath = ''
trackTarget = []
confidence = 0.3
output = False

TRACKER_TYPES = ['Boosting', 
                 'CSRT (recommended)',
                 'KCF (recommended)',
                 'MedianFlow',
                 'MIL',
                 'MOSSE',
                 'TLD',
                 'GOTURN',
                 'Deep Learning (recommended)']

OBJECT_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def call_all():
    global trackTarget
    var_all_none.set("All")
    for var in var_list_target_check:
        var.set(True)
    trackTarget = list(OBJECT_CLASSES)

def call_none():
    global trackTarget
    var_all_none.set("None")
    for var in var_list_target_check:
        var.set(False)
    trackTarget = []

def call_check_target(index):
    global trackTarget
    def toggle():
        # Check target_selected && Update list_trackTarget
        if var_list_target_check[index].get():
            if OBJECT_CLASSES[index] not in trackTarget:
                trackTarget.append(OBJECT_CLASSES[index])
        else:
            if OBJECT_CLASSES[index] in trackTarget:
                trackTarget.remove(OBJECT_CLASSES[index])
        
        # Update state: var_all_none && Radiobutton
        if not any(var.get() for var in var_list_target_check):
            call_none()
        elif len(trackTarget) == len(OBJECT_CLASSES):
            call_all()
        else:
            var_all_none.set(0)  # set Neutral
    return toggle

def call_confidence(value_confidence):
    global confidence
    confidence = float(value_confidence)

def call_video_file_selector():
    global videoPath
    videoPath = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        filetypes=[('Video files', ('.mp4', '.avi'))]
    )
    if videoPath:
        box_videoFileSelector.delete(0, tk.END)
        box_videoFileSelector.insert(0, videoPath)
    else:
        box_videoFileSelector.delete(0, tk.END)
        box_videoFileSelector.insert(0, 'No video file selected')

def call_output():
    global output
    output = var_output.get()

def call_confirm():
    global trackerType
    trackerType = var_tracker_selected.get()
    
    if videoPath == 'No video file selected' or videoPath == '':
        messagebox.showinfo("Info", "Please select a video")
    else:
        mainw.destroy() 
        mainw.quit()

        if trackerType == 'Deep Learning (recommended)':
            print("trackerType: ", trackerType)
            print("trackTarget: ", trackTarget)
            print("confidence: ", confidence)
            print("videoPath: ", videoPath)
            print("output: ", output)

            from MOT_DL import mot_DL
            mot_DL.getValue(trackerType, trackTarget, confidence, videoPath, output)
            mot_DL.mot()
        else:
            print("trackerType: ", trackerType)
            print("videoPath: ", videoPath)
            print("output: ", output)

            from MOT_OpenCV import mot_OpenCV
            mot_OpenCV.getValue(trackerType, videoPath, output)
            mot_OpenCV.mot()

def call_reset():
    global trackerType
    trackerType = ''
    var_tracker_selected.set(TRACKER_TYPES[1])

    call_none()

    global confidence
    confidence = 0.3
    var_confidence.set(0.3)

    global videoPath
    videoPath = ''
    box_videoFileSelector.delete(0, tk.END)

    global output
    output = False
    var_output.set(output)




####################################################################
#                         Prgm Principal
####################################################################

if __name__ == "__main__":

    #Debut IHM ----------------------------------------------------------
    mainw = tk.Tk()
    mainw.title('Multi Object Tracking')
    # mainw.geometry('1000x125')
    mainw.resizable(False, False)

    # Define Widgets
    # txt
    label_trackerTypeSelector = tk.Label(mainw, text='Tracker Type Selector:', font=("Arial", 9, "bold"), anchor='w')
    label_videoFileSelector = tk.Label(mainw, text='Video File Selector:', font=("Arial", 9, "bold"), anchor='w')
    label_config = tk.Label(mainw, text = "Configuration:", font=("Arial", 9, "bold"), anchor='e')
    label_target = tk.Label(mainw, text = "Target Object:", anchor='e')
    label_confidence = tk.Label(mainw, text = "Confidence:", anchor='e')
    label_output = tk.Label(mainw, text = "Output:", font=("Arial", 9, "bold"), anchor='w')

    box_videoFileSelector = tk.Entry(mainw, width=80)

    # optionMenu
    var_tracker_selected = tk.StringVar(mainw)
    var_tracker_selected.set(TRACKER_TYPES[1])
    om_trackerTypeSelector = tk.OptionMenu(mainw, var_tracker_selected, *TRACKER_TYPES)
    om_trackerTypeSelector["width"] = 80

    # bouton
    b_videoSelect = tk.Button(mainw, text='Select', width=5, command=call_video_file_selector)
    b_confirm = tk.Button(mainw, text='Confirm', font=("Arial", 9, "bold"), command=call_confirm)
    b_reset = tk.Button(mainw, text='Reset', font=("Arial", 9, "bold"), command=call_reset)
    
    # radioButton
    var_all_none = tk.StringVar(mainw, "None")
    frame_all_none = tk.Frame(mainw)
    b_all = tk.Radiobutton(frame_all_none, text = "All", variable= var_all_none, value="All", command=call_all)
    b_none = tk.Radiobutton(frame_all_none, text = "None", variable= var_all_none, value="None", command=call_none)
    b_all.grid(row=1, column=1)
    b_none.grid(row=1, column=2)

    # checkButton
    var_list_target_check = [tk.BooleanVar(mainw, False) for i in OBJECT_CLASSES]
    frame_target_check = tk.Frame(mainw)
    list_target_checkbuttons = []
    for i, (label, var) in enumerate(zip(OBJECT_CLASSES, var_list_target_check)):
        row = i // 6 # (divide ->) round down the whole number, if 6 checkbuttons per row
        col = i % 6 # (divide ->) take the reminder
        b_target_check = tk.Checkbutton(frame_target_check, text=label, variable=var, command=call_check_target(i))
        b_target_check.grid(row=row, column=col, sticky='w') # 'w': west-aligned (-> left-aligned)
        list_target_checkbuttons.append(b_target_check)
    var_output = tk.BooleanVar(mainw, False)
    b_output = tk.Checkbutton(mainw, text="A new video file will be automatically generated and saved in the path of the original video selected.", variable=var_output, command=call_output)

    # scale
    var_confidence = tk.DoubleVar(mainw, 0.3)
    scale_confidence = tk.Scale(mainw, from_=0, to=1, orient=tk.HORIZONTAL, variable=var_confidence, command=lambda value: call_confidence(value), resolution=0.1, length=600)


    # Layout using the 'grid' method
    mainw.grid_columnconfigure(0, weight=1)
    mainw.grid_columnconfigure(1, weight=1)

    # ROW11, COLONE7 (ingore row/col_0)
    # row1: TrackerType Selector
    label_trackerTypeSelector.grid(row=1, column=1)
    om_trackerTypeSelector.grid(row=1, column=2, columnspan=6)

    def toggle_config():
        # add row2: Configuration
        if var_tracker_selected.get() == "Deep Learning (recommended)":
            label_config.grid(row=2, column=1)
            
            # add row3: Target Object - ALL/NONE Check
            label_target.grid(row=3, column=1)
            frame_all_none.grid(row=3, column=2, columnspan=6)
            # b_all.grid(
            # add row4-7: Object Classes
            frame_target_check.grid(row=4, column=2, columnspan=6)
            # for i, b_target_check in enumerate(list_target_checkbuttons):
            #     b_target_check.grid(row=4 + (i//6), column=2 + (i%6)*2)
            # add row8: Confidence
            label_confidence.grid(row=8, column=1)
            scale_confidence.grid(row=8, column=2,columnspan=6)
        else:
            label_config.grid_remove()
            label_target.grid_remove()
            frame_all_none.grid_remove()
            frame_target_check.grid_remove()
            # for b_target_check in list_target_checkbuttons:
            #     b_target_check.grid_remove()
            label_confidence.grid_remove()
            scale_confidence.grid_remove()
    var_tracker_selected.trace_add("write", lambda *args: toggle_config())

    # row2 -> row9: Video File Selector
    label_videoFileSelector.grid(row=9, column=1)
    box_videoFileSelector.grid(row=9, column=2, columnspan=5)
    b_videoSelect.grid(row=9, column=7)

    # row3 -> row10: Output Check
    label_output.grid(row=10, column=1)
    b_output.grid(row=10, column=2, columnspan=6)

    # row4 -> row11: Confirm / Reset
    b_confirm.grid(row=11, column=2)
    b_reset.grid(row=11, column=6)

    # Start the GUI event loop
    mainw.mainloop()

    # Fin IHM ----------------------------------------------------------