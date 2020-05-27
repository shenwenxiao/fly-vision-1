#!/usr/bin/env python
import sys
import os
path = sys.path[0]
path = path[0:-5] + 'third-party/pysot/'
sys.path.append(path)
print(path)
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import Pose
import os
import argparse
from idmanage import readid
import time

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
def draw_circle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, init, flag, iamge, start

    if 1:
        if event == cv2.EVENT_LBUTTONDOWN and flag == 1:
            drawing = True
            x1, y1 = x, y
            x2, y2 = -1, -1
            flag = 2
            
            init = False    
        x2, y2 = x, y
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
    path = sys.path[0]
    path = path[0:-5] + 'third-party/pysot/'
    cfg.merge_from_file(path + '/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    model = ModelBuilder()
    #model = load_pretrain(model, '/home/develop/ros/src/fly-vision-1/third-party/pysot/pretrained/model.pth').cuda().eval()
    pre = torch.load(path + 'pretrained/model.pth')
    
    model.load_state_dict(pre)
    model.cuda().eval()
    tracker = build_tracker(model)
    
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
                init_rect = np.array([x1, y1, x2-x1, y2-y1])
                tracker.init(image, init_rect)
                
                start = True
                flag_lose = False
                continue
                
            if start is True:
            
                outputs = tracker.track(image)
                bbox = list(map(int, outputs['bbox'])) 
                
                
                res = [int(l) for l in bbox]
                cv2.rectangle(image, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 2)
                pose.position.x = (bbox[0]+bbox[2]/2-image.shape[1]/2) / (image.shape[1]/2)
                pose.position.y = (bbox[1]+bbox[3]/2-image.shape[0]/2) / (image.shape[0]/2)
                cv2.putText(image, str(outputs['best_score']), (res[0] + res[2], res[1] + res[3]), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,0), 1)
                pose.position.z = 1
                if outputs['best_score'] < 0.5:
              
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

