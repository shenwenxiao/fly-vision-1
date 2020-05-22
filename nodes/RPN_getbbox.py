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
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import Pose
from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from idmanage import readid
import time
import iq

def getbbox(data):
    global x1, y1, x2, y2, id_frame, init, start
    id_frame = int(data.position.x)  
    
    x1 = int(data.orientation.x)
    y1 = int(data.orientation.y)
    x2 = int(data.orientation.z)
    y2 = int(data.orientation.w)
    
    init = True   
    start = False
    print([id_frame, x1, y1, x2, y2])
def getwebim(data):
    global image, getim
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = cv_image   
    getim = True
    #print(getim)
   
    
def showImage():
    global x1, y1, x2, y2, init, image, getim, start, id_frame
    init = False
    rospy.init_node('RPN_getbbox', anonymous=True)
    rate = rospy.Rate(80)
    
    id_frame = 0
    init = False
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
    
    rospy.Subscriber('/vision/bbox', Pose, getbbox)
    rospy.Subscriber('/camera/rgb/image_raw', Image, getwebim)
    pub = rospy.Publisher('/vision/target', Pose, queue_size=10) 
    queue = iq.ImageQueue()
    queue.open('buffer')
    
    interval = 2
    i = 1
    t = time.time()
    fps = 0
    while not rospy.is_shutdown():
        if getim:
            t1 = time.time()
            idd = readid(image)        
            
            if init:
                if id_frame < idd - 2: 
                    image = queue.find(id_frame + interval) # frame is an cv2 image.
                    idd = id_frame
                    id_frame = id_frame + interval
                    if image is not None :
                        cv2.putText(image, 'read from history', (300,30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (10, 10, 10), 1)
                else:
                    id_frame = idd                
            if image is None :
                getim = False
                continue
                
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
                   
          
            cv2.putText(image, '#'+str(idd), (30,30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (10, 10, 10), 1)
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
            cv2.putText(image, 'fps='+str(int(fps)), (200,30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (10, 10, 10), 1)
            
            cv2.imshow('image', image)
            cv2.waitKey(1)
            getim = False

        rate.sleep()
'''


def showImage():
    
    global x1, y1, x2, y2, drawing, init, flag, image, getim, start
    rospy.init_node('RPN_getbbox', anonymous=True)
    
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
'''    
if __name__ == '__main__':
    showImage()


