# fly-vision

## 项目简介

基于ROS(Robot Operating System)框架一个物体检测与物体追踪的项目，目前处于持续开发阶段

## 环境配置

当前项目的运行环境为Ubuntu 16.04 LTS + ROS Kinetic Kame + OpenCV 3.3.1，请确保环境安装正确

硬件配置需要USB接口的单目摄像头

### 1 安装ROS Kinetic Kame

请参考官方安装步骤：http://wiki.ros.org/cn/kinetic/Installation/Ubuntu 

ROS其他官方教程：http://wiki.ros.org/cn/ROS/Tutorials

### 2 安装OpenCV 3.3.1

ROS中已经集成了OpenCV库和相关接口功能包，使用以下命令即可安装：

```shell
$ sudo apt-get install ros-kinetic-vision-opencv libopencv-dev python-opencv
```

### 3 安装Qt Creator

Qt Creator是支持ROS编程的IDE之一，更多支持的IDE请参考官方文档：http://wiki.ros.org/IDEs

本项目可以在命令行下运行，不喜欢使用IDE的可以略过这部分内容

我们使用的是ROS Qt Creator Plug-in，下载地址：https://ros-qtc-plugin.readthedocs.io/en/latest/_source/How-to-Install-Users.html#qt-installer-procedure，这里选择的是推荐的 Xenial Online Installer，在下载目录下输入以下代码进入图像界面并完成安装

```shell
$ chmod +x qtcreator-ros-xenial-latest-online-installer.run
$ ./qtcreator-ros-xenial-latest-online-installer.run
```

完成安装后在Desktop文件夹中输入以下代码编辑QtProject-qtcreator-ros-latest.desktop（找不到的可以使用搜索搜寻.desktop文件）

```shell
$ sudo gedit QtProject-qtcreator-ros-latest.desktop 
```

修改Exec行，添加bash -i -c，保证Qt能加载ROS环境变量，例如

```shell
[Desktop Entry]
Type=Application
Exec=bash -i -c /home/lswhao6/QtCreator/latest/bin/qtcreator-ros
Path=/home/lswhao6/QtCreator/latest
Name=Qt Creator (4.8.0)
GenericName=The IDE of choice for Qt development.
GenericName[de]=Die IDE der Wahl zur Qt Entwicklung
Icon=QtProject-qtcreator
Terminal=false
Categories=Development;IDE;Qt;
MimeType=text/x-c++src;text/x-c++hdr;text/x-xsrc;application/x-designer;application/vnd.qt.qmakeprofile;application/vnd.qt.xml.resource;text/x-qml;text/x-qt.qml;text/x-qt.qbs;
```

### 3 创建ROS工作空间

##### 3.1 命令行方式创建

请参考官方步骤：http://wiki.ros.org/catkin/Tutorials/create_a_workspace

##### 3.2 使用Qt Creator创建

打开Qt Creator，选择 + New Project

![1556240256544](https://github.com/fly-vision/fly-vision/blob/master/images/1556240256544.png)

在Other Project中选择ROS Workspace

![1556239188109](https://github.com/fly-vision/fly-vision/blob/master/images/1556239188109.png)

输入工作空间的名称，Distribution选择kinetic，Build System选择CatkinMake，选择工作空间的路径

![1556240856376](https://github.com/fly-vision/fly-vision/blob/master/images/1556240856376.png)

选择版本管理器后点击Finish，出现如下画面表示创建完成

![1556241099975](https://github.com/fly-vision/fly-vision/blob/master/images/1556241099975.png)

### 4 导入项目

#### 在工作空间的src目录下输入以下命名将所有文件导入

```shell
$ git clone git@github.com:fly-vision/fly-vision.git
```

## 运行步骤

### 1 命令行方式

#### 1.1 构建项目

在工作空间目录下使用以下命令

```shell
$ catkin_make
```

#### 1.2 运行

构建完成后使用以下命令可以运行任意node,其中node的名字为项目结构中nodes文件夹下任意node.cpp

```shell
$ rosrun fly-vision node
```

### 2 使用Qt Creator

#### 2.1 构建项目

可以在Projects中看到Build Settings，点击左下角的”锤子“按钮即可构建

![1556246301460](https://github.com/fly-vision/fly-vision/blob/master/images/1556246301460.png)

#### 2.2 运行

可以在Prejects中看到Run Settings，选择Add Run Step可以添加可运行的node，点击右下角的绿色“三角”按钮即可运行node

![1556246538078](https://github.com/fly-vision/fly-vision/blob/master/images/1556246538078.png)

## 模块介绍

### 1 node功能

web_cam：从摄像头中读取图片帧，发布图片帧

kcf_tracker：订阅图片帧，可以鼠标设置追踪框，显示实时追踪效果

add_id：订阅图片帧，为图片添加id，发布添加id的图片帧

read_id：订阅图片帧，从已添加id的图片中读取id

GroundStation_demo：地面站demo

add_memory：将图片帧存入内存

read_memory：将图片帧从内存中取读

test_memory：测试图片存储

### 2 第三方库功能

id_management：id编码和id解码管理

imagequeue：图像队列，用于图像与内存间的存取操作

kcftracker：kcf：追踪算法实现类