# Data-augmentation-for-LiDAR-object-detection

## object_sample.py

This is an implementation of the paper "Part-Aware Data Augmentation for 3D Object Detection in Point Cloud" (https://arxiv.org/abs/2007.13373). An object is extracted from a point cloud and then points are removed from alternate columns and rows of the point cloud grid. 

Original point cloud of object (Left), Downsampled point cloud of object (Right)
![alt text](https://github.com/siddharth130500/Data-augmentation-for-LiDAR-object-detection/blob/main/PA-aug.png?raw=true)

## object_place.py

This is an implementation of the paper "Context-Aware Data Augmentation for 3D Object Detection in Point Cloud" (https://arxiv.org/pdf/2211.10850.pdf). An object is extracted from a point cloud, and it is then placed at a suitable position in a target point cloud in Range-view projection.

![alt text](https://github.com/siddharth130500/Data-augmentation-for-LiDAR-object-detection/blob/main/CA-aug.png?raw=true)
