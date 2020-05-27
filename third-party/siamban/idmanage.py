import cv2.aruco as aruco
def readid(frame):

    channel_0 = frame[0:24, 0:24, 0]
    channel_1 = frame[0:24, 0:24, 1]
    channel_2 = frame[0:24, 0:24, 2]

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters_create()

    corners, id_1, rejectedImgPoints = aruco.detectMarkers(channel_0, aruco_dict, parameters=parameters)
    if len(id_1)>0:
        corners, id_k, rejectedImgPoints = aruco.detectMarkers(channel_1, aruco_dict, parameters=parameters)
        if len(id_k)>0:
            corners, id_m, rejectedImgPoints = aruco.detectMarkers(channel_2, aruco_dict, parameters=parameters)
            if len(id_m)>0:
                
                idd = id_m[0][0] * 1000 * 1000 + id_k[0][0] * 1000 + id_1[0][0]
            else:
                idd = -1
        else:
            idd = -1
    else:
        idd = -1

    
    return idd;
