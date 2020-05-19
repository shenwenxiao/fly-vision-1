#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Pose.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

#include "id_management.h"

using namespace std;
using namespace cv;


image_transport::Publisher image_pub;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "web_cam");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ros::Rate loop_rate(30);
    // 在这里修改发布话题名称
    image_pub = it.advertise("/camera/rgb/image_raw", 1);

    // 用系统默认驱动读取摄像头0，使用其他摄像头ID，请在这里修改
    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat frame;

    sensor_msgs::ImagePtr msg;
    int id = 0;
    while (ros::ok())
  {
        cap >> frame;
        if (!frame.empty())
        {
            //添加ID
            frame = IDManagement::add_id(id, frame);
            id = id+1;
            // 设置图像帧格式->bgr8
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            // 将图像通过话题发布出去
            image_pub.publish(msg);
        }
        ros::spinOnce();
        // 按照设定的帧率延时，ros::Rate loop_rate(30)
        loop_rate.sleep();
    }

    cap.release();
}

