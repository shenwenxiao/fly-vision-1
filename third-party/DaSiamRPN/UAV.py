# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
import os
import time
# load net
import random
def test(score):

    net = SiamRPNvot()
    net.load_state_dict(torch.load('/home/traker_hao/code/learn/train_RPN/model/30.model'))
    net.eval().cuda()

    version_name='jiasu'

    sequence_path='/media/traker_hao/data/dataset/UAV1/sequences'
    init_path='/media/traker_hao/data/dataset/UAV1/annotations'
    result_path='/home/traker_hao/result/visdrone/'+version_name
    if os.path.exists(result_path) is False:
        os.mkdir(result_path)
    
    sequence_names = os.listdir(sequence_path)
    random.shuffle(sequence_names)
    #sequence_names.sort()
    i=0
    for sequence_name in sequence_names:
        print(sequence_name)
        #if sequence_name != 'Suv':
            #continue
        #sequence_name='uav0000054_00000_s'
        imagenames = os.listdir(sequence_path+'/'+sequence_name)
        imagenames.sort()
        print(i)
        i=i+1
        print(sequence_path+'/'+sequence_name)
        f = open(result_path+'/'+sequence_name+'_'+version_name+'.txt','w')
        inited = False
        fp=open(init_path+'/'+sequence_name+'.txt')
        j=0
        for imagename in imagenames:
            j=j+1
            image = cv2.imread(sequence_path+'/'+sequence_name+'/'+imagename)
            #init the tracker
            if inited is False:
                data = fp.readline()
                data = data.strip('\n') 
                data = data.split(',')   
                [cx, cy, w, h]=(int(data[0])+int(data[2])//2, int(data[1])+int(data[3])//2, int(data[2]), int(data[3]))
                #f.write(str(annos[0]['bbox'][0])+','+str(annos[0]['bbox'][1])+','+str(annos[0]['bbox'][2])+','+str(annos[0]['bbox'][3])+','+str(1.00)+'\n')
                f.write(data[0]+','+data[1]+','+data[2]+','+data[3]+'\n')
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                state = SiamRPN_init(image, target_pos, target_sz, net)
                inited = True
                
                cv2.rectangle(image, (int(cx)-int(w)//2, int(cy)-int(h)//2), (int(cx) + int(w)//2, int(cy) + int(h)//2), (0, 255, 0), 3)
                cv2.putText(image, sequence_name, (50, 50), 0, 5e-3*200, (0,255,0), 2)
                cv2.putText(image, 'initing...', (100, 100), 0, 5e-3*200, (0,255,0), 2)
                image2=cv2.resize(image,(960,540))
                cv2.imshow('aa2', image2)
                cv2.waitKey(1)
            else:

                data = fp.readline()
                data = data.strip('\n') 
                data = data.split(',')  
                try: 
                    truth=(int(data[0]),int(data[1]),int(data[0])+int(data[2]),int(data[1])+int(data[3]))
                except:
                    truth = [0, 0, 0, 0]

                #update the tracker
                #print([cx, cy, w, h])
                tic = cv2.getTickCount()
                t1 = time.time()
                state = SiamRPN_track(state, image)  # track
                #state['target_sz'] = np.array( [int(data[2]), int(data[3])] )
                
                toc = (cv2.getTickCount()-tic) /cv2.getTickFrequency()  
                #print(1/toc) 
                #mytracker.target_sz = np.array([int(truth[2]),int(truth[3])])
                res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                res = [int(l) for l in res]
                cv2.rectangle(image, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 2)
            
                #visualize the result
                
                cv2.rectangle(image, (int(truth[0]), int(truth[1])), (int(truth[2]), int(truth[3])), (0, 255, 0), 2)
                #mytracker.target_sz=np.array([int(data[2]),int(data[3])])
                #cv2.putText(image, str(iou), (res[0] + res[2], res[1] + res[3]), 0, 5e-3*200, (0,255,0), 2)
            cv2.putText(image, sequence_name, (50, 50), 0, 5e-3*200, (0,255,0), 2)
            image2=cv2.resize(image,(960,540))
            cv2.imshow('aa2', image2)
            if cv2.waitKey(1)==97:
                break
            #if j>209:
                #cv2.waitKey(0)
    
        f.close()

test(100)



