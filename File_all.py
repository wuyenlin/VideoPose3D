#!/usr/bin/python3

import os, json, csv
from numpy.linalg import norm
import numpy as np 
import cv2 as cv

class All:
    def __init__(self, S, action, num, seq):
        self.S = S
        self.action = action
        self.num = num
        self.seq = seq

        self.avi_path = os.path.join("dataset/", "S{}/Image_Data/{}_{}_(C{}).avi".format(self.S, self.action, self.num, self.seq))

    def __del__(self):
        print("Killed")

    def save_cropped(self, save_img=True):
        cap = cv.VideoCapture(self.avi_path)
        if (cap.isOpened()==False):
            print("Error opening the video file.")
        k = 0
        while (cap.isOpened()):
            if k > int(cap.get(cv.CAP_PROP_FRAME_COUNT)):
                print("Done!")
                break
            ret, frame = cap.read()
            if ret:

                folder = os.path.join("dataset/", "S{}/Image/{}".format(self.S, self.action))
                filename = os.path.join("dataset/", "S{}/Image/{}/{}_{}_(C{})_{:04}.jpg".format(self.S, self.action, self.action, self.num, self.seq, k))
                try:
                    os.mkdir(folder)
                except FileExistsError:
                    pass

                if save_img:
                    try:
                        cv.imwrite(filename, frame)
                        print("Saving frame {},{},{},{},{}".format(self.S,self.action,self.num,self.seq,k))
                    except FileExistsError:
                        print("File exists")

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            k += 1
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__": 

    c = All(1, "Box", 1, 1)
    c.save_cropped()