import numpy as np

proj_H, proj_W = 64, 1920

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

def range_proj(pc):
    fov_up = 3.0 / 180.0 * np.pi  # field of view up in rad
    fov_down = -25.0 / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(pc[:, :3], 2, axis=1)  # [m,]: range (depth)

    scan_x = pc[:, 0]  # [m,]
    scan_y = pc[:, 1]  # [m,]
    scan_z = pc[:, 2]  # [m,]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)  # [m,]
    pitch = np.arcsin(scan_z / depth)  # [m,]

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(np.minimum(proj_W - 1, np.nan_to_num(proj_x)), 0).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(np.minimum(proj_H - 1, np.nan_to_num(proj_y)), 0).astype(np.int32)

    proj_range = np.full((proj_H, proj_W, 2), -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = np.concatenate((depth.reshape((depth.shape[0],1)),pc[:,4].reshape(pc.shape[0],1)),axis=1)

    return proj_range

def display_range_proj(proj_pc):
    img = np.zeros((proj_H, proj_W))
    img[np.where(proj_pc[:, :, 1] > 0)] = 1
    img[np.where(proj_pc[:, :, 1] == 0)] = 0
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('brg', lut=2)
    colored_image = cm(img)

    from PIL import Image
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(
        '/home/sid/pillarnext/Density_opt/Data Augmentation/test3.png')


if __name__ == "__main__":


    pt_path = '/home/sid/kitti_detection/training/velodyne_original/000008.bin'
    label_path = '/home/sid/kitti_detection/training/label_2/000008.txt'
    calib_path = label_path.replace('label_2', 'calib')

    pts = np.fromfile(pt_path, dtype=np.float32).reshape(-1,4)

    calib = read_calib(calib_path)
    labels = read_label(label_path, calib)

    label = labels[0]  # selecting a random object from the point cloud

    # Extracting object from the point cloud
    mask_x = (pts[:,0] >= label[0][0]-label[1][2]/2) & (pts[:,0] <= label[0][0]+label[1][2]/2)
    mask_y = (pts[:,1] >= label[0][1]-label[1][0]/2) & (pts[:,1] <= label[0][1]+label[1][0]/2)
    mask_z = (pts[:,2] >= label[0][2]-label[1][1]/2) & (pts[:,2] <= label[0][2]+label[1][1]/2)
    mask = mask_x & mask_y & mask_z
    obj1 = pts[np.where(mask)]

    # The target point cloud should be stored in the format (m,5), where the last coordinate is 1 for non-ground and 0 for ground.
    pc = np.fromfile('/home/sid/pillarnext/Density_opt/Data Augmentation/pts.bin', dtype=np.float32).reshape(-1,5)

    proj_pc = range_proj(pc)

    V = np.ones((proj_W,))*np.inf
    for i in range(proj_W):
        idx = np.where(proj_pc[:,i,1]==1)
        if idx[0].shape[0]==0:
            continue
        V[i] = np.min(proj_pc[idx,i,0])

    obj1 = np.concatenate((obj1,np.ones((obj1.shape[0],1))), axis=1)
    proj_obj = range_proj(obj1)

    lg = np.max(np.where(proj_obj[:,:,1]>-1)[1]) - np.min(np.where(proj_obj[:,:,1]>-1)[1]) + 1
    Vr = np.zeros((lg,))
    itr = 0
    prev_i = -1
    for i in range(proj_W):
        cnt = np.where(proj_obj[:, i, 1] > -1)[0].shape[0]
        if cnt>0:
            if prev_i>-1:
                itr += i - prev_i
            Vr[itr] = cnt
            prev_i = i
    ng = np.sum(Vr)

    rbox = np.linalg.norm(label[0])
    lbox = label[1][2]

    threshold = 0.9999
    j_final = -1  # Column where we finally want to insert the object

    for j in range(proj_W-lg):
        Mj = V[j:j+lg] > rbox + lbox/2
        rj = np.dot(Mj,Vr)/ng
        if rj>threshold:
            j_final = j
            break

    # shifting the object
    proj_obj = np.roll(proj_obj, j_final - np.min(np.where(proj_obj[:,:,1]>-1)[1]) + 1, axis=1)
    # display_range_proj(proj_obj)

    proj_pc[np.where(proj_obj[:,:,1]>-1)] = proj_obj[np.where(proj_obj[:,:,1]>-1)]
    display_range_proj(proj_pc)
