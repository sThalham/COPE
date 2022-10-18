<h1 align="center">
COPE: End-to-end trainable Constant Runtime Object Pose Estimation
</h1>

<div align="center">
<h3>
<a href="https://github.com/sThalham">Stefan Thalhammer</a>,
<a href="https://github.com/tpatten">Timothy Patten</a>,
<a href="http://github.com/v4r-tuwien">Markus Vincze</a>,
<br>
<br>
Accepted for publication at WACV: Winter Conference on Applications in Computer Vision, 2023, algorithms track
<br>
<br>
<a href="https://arxiv.org/pdf/2208.08807.pdf">[Paper]</a>
<br>
<br>
</h3>
</div>

![6D pose and Detections on multiple datasets](images/hl_mult_data.png)

# Citation
Please cite the paper if you are using the code:

```
@inproceedings{thalhammer2023cope,
title= {COPE: End-to-end trainable constant runtime object pose estimation}
author={S. {Thalhammer} and T. {Patten} and M. {Vincze}},
journal={arXiv preprint arXiv:2208.08807},
year={2022}}
```

# Installation

- Tensorflow and Keras 2.7.4 need to be installed. 
- Results presented in the paper are generated using NVIDIA CUDA 11.6.


```
git clone https://github.com/sThalham/COPE.git
python3 -m pip install opencv-python==4.4.0.40
python3 -m pip install pillow
python3 -m pip install matplotlib
python3 -m pip install transforms3d
python3 -m pip install glumpy
python3 -m pip install open3d-python
python3 -m pip install PyOpenGL
python3 -m pip install imgaug
```

Alternatively, use the provided Dockerfile to deploy a Docker container that satisfies the version requirements.

The Dockerfile provides means to built a testing and training environment.
Requirements are:
 - docker >= 19.03

Container without GPU support:
```
sh startCont\_cpu.sh
```

Container with GPU support (requires NVIDIA GPU and driver to be installed):
```
sh startCont\_gpu.sh
```

# Training and Evaluation

Training data used for all the experiments in the manuscript is obtained from the [BOP-challenge] (https://bop.felk.cvut.cz/datasets/). Training is done using "PBR-BlenderProc4BOP training images" of each individual dataset. Results are provided on the specific test sets for the challenge "BOP'19/20 test images"
To run evaluation using our provided models the data needs to be downloaded and converted to run with our data loading scheme.

```
cd /path/to/cope
```

- prepare BOP data for compatibility with data loaders:
```
python annotation\_scripts/annotate\_BOP.py <BOP_set> </path/to/bop_datasets> </path/to/target/for/dataset>
```
where <BOP_set> is the sub directory of the datasets, e.g. "train\_pbr".

- Training
from the base directory of COPE run (basic usage):
```
python cope/bin/train.py <dataset> </path/to/dataset>
```

- Testing
from the base directory of COPE run:
```
python cope/bin/evaluate.py <dataset> </path/to/dataset> </path/to/model> --convert-model
```

# Models for reproducing the paper results

Trained model for SOA comparison on LM-O and the DR-PC ablation on LM:
[LM/LM-O weights](https://drive.google.com/file/d/1K3tNKV2dV9QOBGBExbkVXRds1ziGoNYM/view?usp=sharing)

Trained model for SOA comparison on IC-BIN:
[IC-BIN weights](https://drive.google.com/file/d/13RoRxlIopBUHMeg0enHJSDsmB2sQocEG/view?usp=sharing)

# Notes:
- This branch is stale and only meant do provide the code used for generating the results of the paper
- maintained branch at [https://github.com/sThalham/COPE/tree/cope\_clean](https://github.com/sThalham/COPE/tree/cope_clean)
