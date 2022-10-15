# tensorflow
FROM tensorflow/tensorflow:2.7.4-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    tmux \
    vim \
    libgl1-mesa-glx\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# uninstall Keras
RUN pip install opencv-python==4.4.0.40
RUN pip install progressbar2
RUN pip install cython
RUN pip install pillow
RUN pip install matplotlib
RUN pip install transforms3d
RUN pip install glumpy
RUN pip install open3d==0.13.0
RUN pip install PyOpenGL
RUN pip install imgaug
RUN pip install pyyaml==5.4.1

RUN git clone https://github.com/sThalham/COPE.git /cope

# Go to pix2pix root
WORKDIR /cope

