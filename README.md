# FaceSpot

## Overview

A DeepFake detection program created by training the MesoNet architecture (https://github.com/dariusaf/mesonet) with negative samples generated which simulates the resolution inconsistencies (https://github.com/danmohaha/CVPRW2019_Face_Artifacts) created as a by-product of the current DeepFake algorithm.

## Requirements

- Ubuntu 16.04
- Python 2.7 and the following dependencies:
```
NumPy and SciPy
Pillow
Dlib 19.16.0
OpenCV 3.4.0
```
- CUDA 8.0
- cuDNN v6.0
- Tensorflow 1.4.0
- Keras 2.1.5


Additionally, if intending to launch the graphical user interface built using PyQt5, run these commands:

```
apt install -y python-dev

# dependencies for PyQt5
pip install enum34
add-apt-repository ppa:beineri/opt-qt-5.12.0-xenial
apt-get update
apt-get install -y build-essential libgl1-mesa-dev qt512-meta-minimal qt512webengine qt512svg

# install SIP
wget https://www.riverbankcomputing.com/static/Downloads/sip/4.19.14/sip-4.19.14.tar.gz
tar -xvzf sip-4.19.14.tar.gz
cd sip-4.19.14
python configure.py --sip-module=PyQt5.sip
make -j 4
make install

# install PyQt5
wget https://www.riverbankcomputing.com/static/Downloads/PyQt5/5.12/PyQt5_gpl-5.12.tar.gz
tar -xvzf PyQt5_gpl-5.12.tar.gz
cd PyQt5_gpl-5.12
LD_LIBRARY_PATH=/opt/qt512/lib python configure.py --confirm-license --disable=QtNfc --qmake=/opt/qt512/bin/qmake QMAKE_LFLAGS_RPATH=
make -j 4
make install
```

## Preprocess, training and testing paths

By default, the folder structure recognized by the code as-is is as follows:

```
.
├── 8a_mesoinception4.h5 (model weights)
├── imgs (contains all images to be processed into pos- and neg- samples in its root)
|   ├── pos (generated pos- samples)
|   └── neg (generated neg- samples)
├── train_imgs (to be manually filled with pos- and neg- training data)
|   ├── positive
|   └── negative
├── test_imgs (to be manually filled with pos- and neg- training data)
|   ├── positive
|   └── negative
└── dlib_model
    └── shape_predictor_68_face_landmarks.dat
    
```
    
## Workflow

1. Generate positive and negative samples by giving preprocess.py a folder filled with images. All output images will be resized to 256x256px, and negative samples will be applied DeepFake simulation.

2. Manually split generated images into desired train-test ratio, and place respective data into the train_imgs and test_imgs folders, separated by pos- and neg-.

3. Run train.py and test.py.
