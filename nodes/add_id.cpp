#include <ros/ros.h>  
#include <image_transport/image_transport.h>  
#include <cv_bridge/cv_bridge.h>  
#include <sensor_msgs/image_encodings.h>  
#include <geometry_msgs/Pose.h>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <thread>
#include <chrono>
#include <opencv2/aruco.hpp>

#include "id_management.h"

using namespace std;
using namespace cv;

image_transport::Subscriber imageSubscriber_;

int frameWidth_;
int frameHeight_;

std_msgs::Header imageHeader_;
cv::Mat camImageCopy_;
boost::shared_mutex mutexImageCallback_;
bool imageStatus_ = false;
boost::shared_mutex mutexImageStatus_;


bool getImageStatus(void)
{
    boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
    return imageStatus_;
}


void cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ROS_DEBUG("USB image received.");
    cv_bridge::CvImagePtr cam_image;
    try {
        cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        imageHeader_ = msg->header;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if (cam_image) {
        {
            boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
            camImageCopy_ = cam_image->image.clone();
        }
        {
            boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
            imageStatus_ = true;
        }
        frameWidth_ = cam_image->image.size().width;
        frameHeight_ = cam_image->image.size().height;
    }
    return;
}


image_transport::Publisher image_pub;
int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_id");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh); 
    ros::Rate loop_rate(30);
    
    image_pub = it.advertise("/camera/rgb/image_id", 1);
    
    // 接收图像的话题
    imageSubscriber_ = it.subscribe("/camera/rgb/image_raw", 1, cameraCallback);
    const auto wait_duration = std::chrono::milliseconds(2000);
    sensor_msgs::ImagePtr idmsg;
    int id = 0;
    while (ros::ok()){
        while (!getImageStatus()) {
            printf("Waiting for image.\n");
            std::this_thread::sleep_for(wait_duration);
            ros::spinOnce();
        }
        Mat frame;
        {
            boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
            frame = camImageCopy_.clone();
        }

        //添加ID到照片左上角
        frame = IDManagement::add_id(id,frame);
        // 设置图像帧格式->bgr8
        idmsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        // 将图像通过话题发布出
        image_pub.publish(idmsg); 
       	if( getImageStatus() )  id = id+1;
        printf("The ID what I have send is=%d\n", id);
		 
        imshow("add_id", frame);
        waitKey(5);
        
        ros::spinOnce();
        loop_rate.sleep();
    }
		
}

