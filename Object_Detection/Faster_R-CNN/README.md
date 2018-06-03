Faster R-CNN
===============

How to use
-------

1. Clone and follow the README of python implemented Faster R-CNN from https://github.com/rbgirshick/py-faster-rcnn. Install every requirement software including Caffe and package.
2. Put the folder "AirSim_sample_input" in py-faster-rcnn/data. Put "AirSim_COCO.py" in py-faster-rcnn/tools. Creat a new floder named "output" in py-faster-rcnn/data.
3. Replace the PASCAL VOC model with COCO model. See "How to use COCO dataset model" in this file.
4. python AirSim_COCO.py (under python version 2.7)

How to use COCO dataset model
--------- 

1. Download the model from https://dl.dropboxusercontent.com/s/cotx0y81zvbbhnt/coco_vgg16_faster_rcnn_final.caffemodel?dl=0.
2. Put it in py-faster-rcnn/data/faster_rcnn_models/
3. Copy "test.prototxt" from py-faster-rcnn/models/coco/VGG16/faster_rcnn_end2end/ to py-faster-rcnn/models/pascle_voc/VGG16/faster_rcnn_alt_opt. Rename it to "coco_test.pt".
4. Make sure the setting and classes of COCO in "AirSim_COCO.py" is used.

How to get test images
-----------

1. Clone and install the AirSim from https://github.com/Microsoft/AirSim as well as the AirSimNH environment.
2. Put "captureImages.py" and "AirSimClient.py" in AirSim/PythonClient.
3. Run AirSimNH.exe by run.bat to setup the environment.
4. python captureImages.py (under python version 3.6)
5. Enter the coordinate of destination. Then the UAV will fly to it and capture image in AirSim\PythonClient\AirSim_sample_input
6. Copy the folder "AirSim_sample_input" to py-faster-rcnn/data.
