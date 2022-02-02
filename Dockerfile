# tensorflow
FROM tensorflow/tensorflow:2.7.0-gpu

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
RUN pip uninstall --yes keras
RUN pip install opencv-python==4.4.0.40
RUN pip install progressbar2
RUN pip install cython
RUN pip install pillow
RUN pip install matplotlib
RUN pip install transforms3d
RUN pip install glumpy
RUN pip install open3d-python
RUN pip install PyOpenGL
RUN pip install imgaug
RUN pip install pyyaml

RUN git clone https://github.com/sThalham/PyraPoseAF.git /PyraPoseAF

# Go to pix2pix root
WORKDIR /PyraPoseAF


#CMD ["python3", "RetinaNetPose/bin/train.py", "linemod", "data"]
