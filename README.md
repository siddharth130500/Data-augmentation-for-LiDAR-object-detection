# Data-augmentation-for-LiDAR-object-detection

## object_sample.py

This is an implementation of the paper "Part-Aware Data Augmentation for 3D Object Detection in Point Cloud" (https://arxiv.org/abs/2007.13373). An object is extracted from a point cloud and then points are removed from alternate columns and rows of the point cloud grid. 

![alt text](https://github.com/siddharth130500/DGR/blob/main/60_01.png?raw=true)

## object_place.py

This is an implementation of the paper "Context-Aware Data Augmentation for 3D Object Detection in Point Cloud" (https://arxiv.org/pdf/2211.10850.pdf). An object is extracted from a point cloud, and it is then placed at a suitable position in a target point cloud in Range-view projection.

![alt text](https://github.com/siddharth130500/DGR/blob/main/CA-aug.png?raw=true)
