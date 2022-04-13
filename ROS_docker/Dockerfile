FROM nvidia/cuda:10.0-cudnn7-devel
FROM tensorflow/tensorflow:2.1.1-gpu
MAINTAINER Stefan Thalhammer (thalhammer@acin.tuwien.ac.at)
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# ROS
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      ros-melodic-ros-core \
      ros-melodic-libuvc-camera \
      ros-melodic-image-view \
      ros-melodic-cv-bridge \
      ros-melodic-cv-camera \
      ros-melodic-actionlib \
      && rm -rf /var/lib/apt/lists/*

# Linux package install
RUN sed 's/main$/main universe/' -i /etc/apt/sources.list
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated --fix-missing \
      build-essential \
      autoconf \
      automake \
      git \
      wget \
      usbutils \
      unzip \
      vim \
      software-properties-common \
      libxext-dev \
      libxrender-dev \
      libxslt1.1 \
      libxtst-dev \
      libgtk2.0-0 \
      libcanberra-gtk-module \
      tmux \
      g++ \
      openssh-client \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-virtualenv \
      python3-wheel \
      pkg-config \
      libopenblas-base \
      python3-numpy \
      python3-scipy \
      python3-h5py \
      python3-yaml \
      python3-pydot \
      python3-tk \
      libgl1-mesa-glx\
      python-rosinstall \
      python-rosinstall-generator \
      python-wstool \
      python-rosdep \
      python-vcstools \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*
# Fixes shared memory error in docker
RUN echo "export QT_X11_NO_MITSHM=1" >> ~/.bashrc
# Python 3
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-dev \
     python3-numpy \
     python3-pip \
     && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
# install Keras
RUN pip3 install keras==2.2.5
RUN pip3 install opencv-python==3.4.0.14
RUN pip3 install progressbar2
RUN pip3 install keras-resnet
RUN pip3 install cython
RUN pip3 install pycocotools
RUN pip3 install pillow
RUN python3 -m pip install matplotlib
RUN pip3 install transforms3d
RUN pip3 install pyquaternion
RUN pip3 install Geometry
RUN pip3 install glumpy
RUN pip3 install open3d-python
RUN pip3 install pygeometry
RUN pip3 install PyOpenGL
RUN git clone https://github.com/sThalham/RGBDPose.git /RGBDPose
# ROS
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      python-rosinstall \
      python-rosinstall-generator \
      python-wstool \
      python-rosdep \
      python-vcstools \
      && rm -rf /var/lib/apt/lists/*
# bootstrap rosdep
RUN rosdep init && \
    rosdep update
# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python-catkin-tools \
     && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip setuptools
RUN pip3 install --upgrade rospkg catkin_pkg opencv-contrib-python empy
# for ros environments
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# Prepare catkin build
RUN mkdir -p ~/catkin_build_ws/src
# ROS-opencv
RUN /bin/bash -c 'cd ~/catkin_build_ws/src; git clone -b melodic https://github.com/ros-perception/vision_opencv.git'
# Detector ROS messages
RUN /bin/bash -c 'cd ~/catkin_build_ws/src; git clone http://gitlab+deploy-token-26:X91Kzy7k1oCpZyWa1LzD@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
# Detectron ROS wrapper
RUN /bin/bash -c 'cd ~/catkin_build_ws/src; git clone http://gitlab+deploy-token-25:Rr3AfyRtB5TAspwWYYMZ@rgit.acin.tuwien.ac.at/msuchi/detectron2_ros.git'
# Run catkin build
RUN /bin/bash -c  '. /opt/ros/melodic/setup.bash; cd ~/catkin_build_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'
# source the catkin workspace
RUN echo "source ~/catkin_build_ws/devel/setup.bash" >> ~/.bashrc
WORKDIR /root
# setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
