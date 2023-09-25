import pdb

import numpy as np
from tqdm import tqdm
import glob
import os


def read_calib(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    obj = lines[4].strip().split(' ')[1:]
    rect_4x4 = np.eye(4, dtype=np.float32)
    rect_4x4[:3, :3] = np.array(obj, dtype=np.float32).reshape(3, 3)

    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3] = np.array(obj, dtype=np.float32).reshape(3, 4)

    world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)

    return {'world_cam': world_cam}

def read_label(path, calib):

    with open(path, 'r') as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        label = line.strip().split(' ')

        center = np.array(
            [float(label[11]),
             float(label[12]),
             float(label[13]), 1.0])

        points = center @ np.linalg.inv(calib['world_cam'])

        size = [float(label[9]), float(label[8]), float(label[10])]  # w,h,l
        center = [points[0], points[1], size[1] / 2 + points[2]]
        objects.append([center, size, label[0]])

    return objects

if __name__ == "__main__":

    pt_path = '/home/sid/kitti_detection/training/velodyne_original/000008.bin'
    label_path = '/home/sid/kitti_detection/training/label_2/000008.txt'
    calib_path = label_path.replace('label_2', 'calib')

    pts = np.fromfile(pt_path, dtype=np.float32).reshape(-1,4)

    calib = read_calib(calib_path)
    labels = read_label(label_path, calib)

    label = labels[0]

    # Extracting the object out from the point cloud
    mask_x = (pts[:,0] >= label[0][0]-label[1][2]/2) & (pts[:,0] <= label[0][0]+label[1][2]/2)
    mask_y = (pts[:,1] >= label[0][1]-label[1][0]/2) & (pts[:,1] <= label[0][1]+label[1][0]/2)
    mask_z = (pts[:,2] >= label[0][2]-label[1][1]/2) & (pts[:,2] <= label[0][2]+label[1][1]/2)
    mask = mask_x & mask_y & mask_z
    obj1 = pts[np.where(mask)]

    omega = []  # Elevation angle for each point in the object
    for i in range(obj1.shape[0]):
        omega.append(np.arcsin(obj1[i][2] / np.sqrt(obj1[i][0] ** 2 + obj1[i][1] ** 2 + \
                                                       obj1[i][2] ** 2)))
    omega = (np.array(omega)) * 180 / np.pi
    Gv = np.linspace(-24.8, 2, num=64)

    obj_mask = np.ones((obj1.shape[0],))

    for i in (range(omega.shape[0])):
        idx = np.argmin(np.abs(Gv - omega[i]))
        if idx%2:
            obj_mask[i] = 0   # Skipping every alternate beam

    obj1 = obj1[np.where(obj_mask!=0)]

    phi = [] # Azimuthal angle for each point in the object
    for i in range(obj1.shape[0]):
        phi.append(np.sign(obj1[i][1]) * np.arccos(obj1[i][0] / np.sqrt(obj1[i][0] ** 2 + obj1[i][1] ** 2)))
    phi = (np.array(phi)) * 180 / np.pi
    Gh = np.linspace(-180, 180, num=int(360/0.16))

    obj_mask = np.ones((obj1.shape[0],))

    for i in (range(phi.shape[0])):
        idx = np.argmin(np.abs(Gh - phi[i]))
        if idx % 2:
            obj_mask[i] = 0  # Skipping every alternate vertical column of points

    obj1 = obj1[np.where(obj_mask != 0)]

    obj1.astype(np.float32).tofile('/home/sid/obj.bin')
