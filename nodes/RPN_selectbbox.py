#!/usr/bin/env python
import sys
import os
path = sys.path[0]
path = path[0:-5] + 'third-party/DaSiamRPN/'
sys.path.append(path)

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import Pose
from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from idmanage import readid
import time
'''
def draw_circle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, init, flag, iamge

    if init is False:
        #print(init)
        if event == cv2.EVENT_LBUTTONDOWN and flag == 2:
            if drawing is True:
                drawing = False
                x2, y2 = x, y
                init = True
                #flag = 1
                print(init)
                print([x1,y1,x2,y2])

        if event == cv2.EVENT_LBUTTONDOWN and flag == 1:

            drawing = True
            x1, y1 = x, y
            x2, y2 = -1, -1
            flag = 2
        if drawing is True:
            x2, y2 = x, y
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if event == cv2.EVENT_MBUTTONDOWN:
        flag = 1
        init = False
        x1, x2, y1, y2 = -1, -1, -1, -1
'''

def draw_circle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, init, flag, iamge, start

    #if init is False:
    if 1:
        if event == cv2.EVENT_LBUTTONDOWN and flag == 1:
            drawing = True
            x1, y1 = x, y
            x2, y2 = -1, -1
            flag = 2
            
            init = False    
            
        #print(init)
        x2, y2 = x, y
        #if event == cv2.EVENT_LBUTTONDOWN and flag == 2:
            #if drawing is True:
                #x2, y2 = x, y
        if event == cv2.EVENT_LBUTTONUP and flag == 2:
            w = x2-x1
            h = y2 -y1
            if w>0 and w*h>50:
                init = True   
                start = False   
                flag = 1
                drawing = False
                print(init)
                print([x1,y1,x2,y2])
            else:
                x1, x2, y1, y2 = -1, -1, -1, -1
        if drawing is True:
            x2, y2 = x, y
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
        
    
    if event == cv2.EVENT_MBUTTONDOWN:
        flag = 1
        init = False
        x1, x2, y1, y2 = -1, -1, -1, -1
        
def callback(data):
    global image, getim
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = cv_image   
    getim = True
    #print(getim)

def showImage():
    
    global x1, y1, x2, y2, drawing, init, flag, image, getim, start
    rospy.init_node('RPN', anonymous=True)
    
    flag=1
    init = False
    drawing = False
    getim = False
    start = False
    x1, x2, y1, y2 = -1, -1, -1, -1
    flag_lose = False
    count_lose = 0

    print('laoding model...........')
    net = SiamRPNvot()
    net.load_state_dict(torch.load(path + 'SiamRPNVOT.model'))
    net.eval().cuda()
    z = torch.Tensor(1, 3, 127, 127)
    net.temple(z.cuda())
    x = torch.Tensor(1, 3, 271, 271)
    net(x.cuda())
    print('ready for starting!')
    
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
    pub = rospy.Publisher('/vision/target', Pose, queue_size=10) 
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    rate = rospy.Rate(30)
    i = 1
    t = time.time()
    fps = 0
    while not rospy.is_shutdown():
      
        if getim:
            t1 = time.time()
            idd = readid(image)
            
            pose = Pose()
            pose.position.z = 0
            
            if start is False and init is True:
                target_pos = np.array([int((x1+x2)/2), int((y1+y2)/2)])
                target_sz = np.array([int(x2-x1), int(y2-y1)])
                state = SiamRPN_init(image, target_pos, target_sz, net)
                start = True
                flag_lose = False
                continue
                
            if start is True:
            
                state = SiamRPN_track(state, image)  # track              
                res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                res = [int(l) for l in res]
                cv2.rectangle(image, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 2)
                pose.position.x = (state['target_pos'][0]-image.shape[1]/2) / (image.shape[1]/2)
                pose.position.y = (state['target_pos'][1]-image.shape[0]/2) / (image.shape[0]/2)
                cv2.putText(image, str(state['score']), (res[0] + res[2], res[1] + res[3]), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,0), 1)
                pose.position.z = 1
                if state['score'] < 0.5:
                    count_lose = count_lose + 1
                else:
                    count_lose = 0
                if count_lose > 4:
                    flag_lose = True
                    
            if flag_lose is True:
                    cv2.putText(image, 'target is lost!', (200,200), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,0,0), 3)
                    pose.position.z = -1
                   
            if drawing is True:              
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(image, '#'+str(idd), (30,30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 255), 1)
            cx = int(image.shape[1]/2)
            cy = int(image.shape[0]/2)
            cv2.line(image,(cx-20,cy), (cx+20, cy), (255, 255, 255), 2)
            cv2.line(image,(cx, cy-20), (cx, cy+20), (255, 255, 255), 2)
            
            pub.publish(pose)
            
            if start is True:    
               
                i = i + 1
            if i > 5:
                i = 1
                fps = 5 / (time.time()-t)
                t = time.time()
            cv2.putText(image, 'fps='+str(fps), (200,30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 255), 1)
            
            cv2.imshow('image', image)
            cv2.waitKey(1)
            getim = False

        rate.sleep()
    
if __name__ == '__main__':
    showImage()

