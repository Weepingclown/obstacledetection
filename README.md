# ObstacleDetection

I built a package of algorithms to detect and avoid obstacles using a stereo camera.

Obtain disparity from stereo image, preprocessing, and create depth map. Use the Depth map to find the Real wordl coordinates, obtain the ground plane from the image, and calculate the area higher than the floor to eliminate obstacles.

## Maintainer
Joowon Kim (jwkim@incorl.hanyang.ac.kr)

## Table of contents

- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Process](#process)
- [Video](#video)

## Description.

The purpose of this packages(for Ros) is to detect obstacles using the stereo camera
([ZED Stereo camera](https://www.stereolabs.com/)).

## Dependencies

- [ROS Indigo](http://wiki.ros.org/indigo)
- [OpenCV 3.1 ](http://wiki.ros.org/vision_opencv)
- [Zed Stereo camera](https://www.stereolabs.com/)

## Installation
Install Ros-Indigo
[Ros-Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu)  
following this page and setup

Install Opencv3.1
Click this link [http://opencv.org/downloads.html](http://opencv.org/downloads.html)   
and down load VERSION 3.1 OpenCV for Linux/Mac      
extrack the file desired location   
```
cd ~/Download/opencv-3.1.0
sudo cmake .
sudo make -j4 # runs 4 jobs in parallel
sudo make install
```
Verify installation
```
pkg-config --modeversion opencv
```
3.1.0 will be printed

source code download
```
cd ~/catkin_ws/src
git clone https://github.com/joowonkim/obstacledetection.git
```
build:
```
cd ~/catkin_ws
catkin_make
```
Initialize the environment:
```
cd ~/catkin_ws
source devel/setup.bash
```
Run:
```
roscore
# open the new terminal using ctrl+shit tab
rosrun obstacledetection obstacledetection
```
However,"Segmentation fault(core dumped)" is displayed and not working.

Plan to add more work...

## Process
![color image](http://dl.dropbox.com/s/975px17ee8hmvza/process.png)
***
![color image](http://dl.dropbox.com/s/gs4o7lr7kfy6t9v/process2.png)
***
![color image](http://dl.dropbox.com/s/s77277syon4eoae/process3.png)
***
![color image](http://dl.dropbox.com/s/pqjclc080717wfn/process4.png)
***
![color image](http://dl.dropbox.com/s/va1d2kgh8jatba9/process5.png)
***
![color image](http://dl.dropbox.com/s/0cb0qv3cjh1gscn/process6.png)
***
## Video
[![Demo CountPages alpha](http://dl.dropbox.com/s/qq3xlhrwfexeee4/obstacle.gif)](https://www.youtube.com/embed/Q4toI3cEoTk)   
click ! If you want to see the video
