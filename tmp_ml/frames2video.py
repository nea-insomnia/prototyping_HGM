import os
import cv2 
from PIL import Image


def generate_video(sourceDirectory,videoname,fps):
    images = [img for img in os.listdir(sourceDirectory)]                   #breaks if in the sourcedirectory contains other stuff besides images
    images.sort()
    frame = cv2.imread(os.path.join(sourceDirectory, images[0]))
    height, width, layers = frame.shape  
    #TODO: einen Zielordner als zusätzlichen Parameter hinzufügen, in dem das Video abgespeichert wird  
    video = cv2.VideoWriter(videoname, 0x7634706d, fps, (width, height))    # 0x7634706d hab ich aus dem Internet, damit cv2 mp4-videos schreiben kann
    for image in images: 
        video.write(cv2.imread(os.path.join(sourceDirectory, image)))
    cv2.destroyAllWindows() 
    video.release()


if __name__ == '__main__':
    
	# Calling the function
    sourceDirect = '/home/fkoehler/newenv/kreativ/FSTBT-pytorch/FSPBT-Image-Translation-master/data/swap/output'
    videodirect = '/home/fkoehler/newenv/'
    videoname = 'swap10000ep.mp4'
    framerate = 30
    generate_video(sourceDirect,videodirect,videoname,framerate)
    

