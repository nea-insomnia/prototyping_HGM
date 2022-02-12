# Program To Read video and Extract Frames
from pathlib import Path
import cv2
import os
import shutil

def count2name(number,format = ".jpg"):     #pads numbers with leading zeroes up to 4 digits, breaks for numbers >9999, makes sure that sorting them by name returns the intented order
    pad = "000"+str(number)
    return pad[len(pad)-4:len(pad)]+format


# Function to extract frames
def FrameCapture(videoPath,frameDirectory):
    vidObj = cv2.VideoCapture(videoPath)
    if os.path.isdir(frameDirectory):   #make shure that the data_path/input-folder exists and is empty
        shutil.rmtree(frameDirectory)
    os.mkdir(frameDirectory)
    os.chdir(frameDirectory)

    count = 0                       # Used as counter variable
    success, image = vidObj.read()  #success checks whether a frame could be extracted
    while success:
        cv2.imwrite(count2name(count,".jpg"), image)# Saves the frames with frame-count
        count += 1
        success, image = vidObj.read()# vidObj object calls read function extract frames
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    return fps

if __name__ == '__main__':

	# Calling the function
    videopath = 'ohne.mp4'
    imagedirec = Path('data/ohnezumit')/'input_eval'
    framerate = FrameCapture(videopath,imagedirec)
    print(framerate)
