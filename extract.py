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

        self.avi_path = os.path.join("dataset/", "S{}/Image_Data/{}_{}_\(C{}\).avi".format(self.S, self.action, self.num, self.seq))

    def __del__(self):
        print("Killed")

    def save_cropped(self, save_img=True):
        folder = os.path.join("dataset/", "S{}/Image/{}".format(self.S, self.action))
        filename = os.path.join("dataset/", "S{}/Image/{}/{}_{}_\(C{}\)_$frame%04d.jpg".format(self.S, self.action, self.action, self.num, self.seq))
        try:
            
            os.mkdir(folder)
        except FileExistsError:
            pass

        if save_img:
            try:
                os.system("ffmpeg -i {} -r 60 {}".format(self.avi_path, filename))

                print("Saving frame {},{},{},{}".format(self.S,self.action,self.num,self.seq))
            except FileExistsError:
                print("File exists")


if __name__ == "__main__": 
    action_list = ["Walking", "Jog", "ThrowCatch", "Gestures", "Box"]
    for char in [1,2,3]:
        for action in action_list:
            c = All(char, action, 1, 1)
            c.save_cropped()
    # os.system("ffmpeg -i dataset/walking.avi -r 60 dataset/S1/Image/$frame%04d.jpg")