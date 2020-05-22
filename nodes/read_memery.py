#!/usr/bin/env python
import sys
import os
path = sys.path[0]
path = path[0:-5] + 'third-party/DaSiamRPN/'
sys.path.append(path)

import iq
import cv2
import rospy
def readmem():
    
    
    rospy.init_node('readmem', anonymous=True)
    rate = rospy.Rate(50)
    queue = iq.ImageQueue()
    queue.open('buffer')
    #queue.create('buffer', 1000, 1080, 1920)
    i=10
    while not rospy.is_shutdown():

        frame = queue.find(i) # frame is an cv2 image.
       
        if frame is not None :
            
            cv2.imshow('memery', frame)
            cv2.waitKey(1)
        else:
            print('no this frame (id=%d)'%(i))
        i = i + 1
        rate.sleep()
    '''
    i = 100
    while not rospy.is_shutdown():
        try:
            frame = queue.find(i) # frame is an cv2 image.
        except:
            print('no this id=%d frame'%(i))
        i = i + 1
        cv2.imshow('memery', frame)
        cv2.waitKey(1)
        rate.sleep()
    '''  
if __name__ == '__main__':
    readmem()

