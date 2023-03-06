import numpy as np
import os, imageio
from pathlib import Path
import cv2
'''
def load_dtu_data(path, train_scene, mask_path=None):
    imgdir = os.path.join(path, 'image')
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgnames = [f'{int(i.split("/")[-1].split(".")[0]):03d}.png' for i in imgfiles]
   
    masks = None
    if mask_path is not None:
        cat = imgfiles[0].split('/')[3]
        seen_views  = '_'.join(map(str,train_scene))
        maskdir = os.path.join('./data/DTU_mask/', cat, seen_views)
        maskfiles = [os.path.join(maskdir, i) for i in imgnames] 
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]
    
    if mask_path is not None:
        masks = [imread(f)[...,0]/255. for f in maskfiles]
        masks = np.stack(masks, 0)

    cam_path = os.path.join(path, "cameras.npz")
    all_cam = np.load(cam_path)

    focal = 0

    coord_trans_world = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
    coord_trans_cam = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )

    poses = []
    for i in range(num):
        P = all_cam["world_mat_" + str(i)]
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        focal += (K[0,0] + K[1,1]) / 2

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        scale_mtx = all_cam.get("scale_mat_" + str(i))
        if scale_mtx is not None:
            norm_trans = scale_mtx[:3, 3:]
            norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

            pose[:3, 3:] -= norm_trans
            pose[:3, 3:] /= norm_scale

        pose = (
                coord_trans_world
                @ pose
                @ coord_trans_cam
            )
        poses.append(pose[:3,:4])
    
    poses = np.stack(poses)
    print('poses shape:', poses.shape)


    focal = focal / num
    H, W = imgs[0].shape[:2]
    print("HWF", H, W, focal)

    return imgs, poses, [H, W, focal], masks
'''
import numpy as np
import os, imageio
from scipy.spatial.transform import Rotation as R
import re, cv2

def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) #* self.scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * 1.06 #* self.scale_factor
    return intrinsics, extrinsics, [depth_min, depth_max]

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
    depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_NEAREST)  # (600, 800)
    depth_h = depth_h[44:556, 80:720]  # (512, 640)

    return depth_h

def load_dtu_data(basedir, train_view_num = 16):

    root_dir = os.path.dirname(basedir)
    scan = os.path.basename(basedir)
    light_idx = 3

    # opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    imgs = []
    poses = []
    depths_cas = []
    depths = []
    bds = []
    for vid in range(49):
        img_filename = os.path.join(root_dir, f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
        proj_mat_filename = os.path.join(root_dir, f'Depths/Cameras/train/{vid:08d}_cam.txt')
        depth_filename = os.path.join(root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')
        # depth_filename_cas = f'Depths_infer/{scan}/depth_map_{vid:04d}.pfm'
        #depth_filename_cas = f'nerf_dtu_data_depth/{scan}/depth_{vid:04d}.pfm'
        intrinsic, w2c, near_far = read_cam_file(proj_mat_filename)
        intrinsic[:2] *= 4
        imgs += [imageio.imread(img_filename).astype(np.float32) / 255.]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 3] *= 1/200
        pose = np.concatenate([c2w[:, :1], -c2w[:, 1:2], -c2w[:, 2:3], c2w[:, 3:4]], axis=-1)
        poses += [pose]
        #depths_cas += [np.array(read_pfm(depth_filename_cas)[0], dtype=np.float32)]
        depths += [read_depth(depth_filename)/200]

        bds += [near_far[0]/200, near_far[1]/200]

    imgs = np.stack(imgs, axis=0)
    poses = np.stack(poses, axis=0)
    bds = np.stack(bds, axis=0)
    #depths_cas = np.stack(depths_cas, axis=0)
    depths = np.stack(depths, axis=0)

    print('Loaded', basedir, bds.min(), bds.max())

    render_poses = gen_render_path(poses, N_views=3)
    render_poses = np.array(render_poses).astype(np.float32)
    print(intrinsic)
    H, W = imgs[0].shape[:2]
    focal = float(intrinsic[0,0])
    hwf = [H, W, focal]

    return imgs, poses, bds, render_poses, hwf, depths
