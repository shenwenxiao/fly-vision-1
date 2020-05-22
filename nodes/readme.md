1. 从商汤下载网络文件 https://drive.google.com/open?id=1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H
2. 跟踪结果用归一化的格式发送出去，topic名称 '/vision/target'， <geometry_msgs/Pose.h> 格式, Z = 0 表示跟踪没有启动， Z = 1表示跟踪中，  Z=-1表示跟踪器判定自己跟丢了目标

3. 节点RPN_selectbbox是在飞机端的电脑直接框选目标启动跟踪
4. 节点RPN_getbbox是地面站框选目标启动跟踪，暂时定的topic名称 '/vision/bbox'， <geometry_msgs/Pose.h> 格式, Quaternion. x,y,w,z对应左上角坐标和尺度x,y,w,h,  position的X代表帧数。

5. 运行以下节点模拟选择目标，算法从历史帧里读取照片， GroundStation_demo里的第15行宏定义delay表示通信延迟的时间，现在设定的是2秒，根据需要更改
   rosrun fly-vision web_cam;
   rosrun fly-vision add_memery;
   rosrun fly-vision GroundStation_demo;
   rosrun fly-vision RPN_getbbox;

 
