"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
#import matplotlib.image as mpimg
import cv2
#from docopt import docopt
#from IPython.display import HTML, Video
from IPython.display import display, clear_output
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from playsound import playsound
import time
from playsound import playsound
import cv2
import dlib
from scipy.spatial import distance

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        in_lane = self.lanelines.in_lane
        img = self.transform.backward(img)

        #out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img, in_lane
    
    def process_frame(self, frame):
        out_frame, in_lane = self.forward(frame)
        return out_frame, in_lane

'''
    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)
'''

'''
def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']

    findLaneLines = FindLaneLines()
    if args['--video']:
        findLaneLines.process_video(input, output)
    else:
        findLaneLines.process_image(input, output)
'''

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

def main():

 # Create an instance of FindLaneLines
    find_lane_lines = FindLaneLines()
    
    cap = cv2.VideoCapture(2)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    eyes_open = True
    i = 0
    
 # Open camera capture
    cap1 = cv2.VideoCapture(0)  # 0 for default camera, change to a different number for other cameras

    while True:
        ret, frame = cap1.read()
        if not ret:
            break

        # Process frame
        out_frame, in_lane = find_lane_lines.process_frame(frame)

        # Display processed frame
        clear_output(wait=True)
        display(out_frame)

        cv2.imshow("Lane Detection", out_frame)
        
        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []

            for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)
            if EAR<0.1:
                cv2.putText(frame,"DROWSY",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
                cv2.putText(frame,"Are you Sleepy?",(20,400),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                i += 1
                if(i > 7):
                    eyes_open = False
                    break
                print("Drowsy")
            else:
                eyes_open = True
                i = 0
            print(EAR)

        cv2.imshow("Are you Sleepy", frame)
        
        if(in_lane == False or eyes_open == False):
            playsound("DNA.mp3")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()