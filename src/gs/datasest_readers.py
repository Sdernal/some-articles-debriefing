import cv2 as cv
import numpy as np
from typing import NamedTuple, Optional
import os
import json
from .utils import focal2fov, BasicPointCloud
from PIL import Image
from scipy.spatial.transform import Slerp, Rotation


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    exp: np.array
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def readNerfBlendShapeCameras(path, is_eval, is_debug, novel_view, only_head):
    with open(os.path.join(path, "transforms.json"), 'r') as f:
        meta_json = json.load(f)

    test_frames = -1_000
    frames = meta_json['frames']
    total_frames = len(frames)

    if not is_eval:
        print(f'Loading train dataset from {path}...')
        frames = frames[0: (total_frames + test_frames)]
        if is_debug:
            frames = frames[0: 50]
    else:
        print(f'Loading test dataset from {path}...')
        frames = frames[-50:]
        if is_debug:
            frames = frames[-50:]

    cam_infos = []
    h, w = meta_json['h'], meta_json['w']
    fx, fy, cx, cy = meta_json['fx'], meta_json['fy'], meta_json['cx'], meta_json['cy']
    fovx = focal2fov(fx, pixels=w)
    fovy = focal2fov(fy, h)

    for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
        image_id = frame['img_id']
        image_path = os.path.join(path, "ori_imgs", str(image_id) + '.jpg')
        image = np.array(Image.open(image_path))
        if not only_head:
            mask_path = os.path.join(path, "mask", str(image_id + 1) + '.png')
            seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            # Reference MODNet colab implementation
            mask = np.repeat(np.asarray(seg)[:, :, None], 3, axis=2) / 255
        else:
            mask_path = os.path.join(path, "parsing", str(image_id) + '.png')
            seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
            if seg.shape[-1] == 3:
                seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
            else:
                seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
            mask = (seg[:, :, 0] == 0) * (seg[:, :, 1] == 0) * (seg[:, :, 2] == 255)
            mask = np.repeat(np.asarray(mask)[:, :, None], 3, axis=2)

        white_background = np.ones_like(image) * 255
        image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))

        expression = np.array(frame['exp_ori'])
        if novel_view:
            vec = np.array([0, 0, 0.3493212163448334])
            rot_cycle = 100
            tmp_pose = np.identity(4, dtype=np.float32)
            r1 = Rotation.from_euler('y', 15 + (-30) * ((idx % rot_cycle) / rot_cycle), degrees=True)
            tmp_pose[:3, :3] = r1.as_matrix()
            trans = tmp_pose[:3, :3] @ vec
            tmp_pose[0:3, 3] = trans
            c2w = tmp_pose
        else:
            c2w = np.array(frame['transform_matrix'])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                    image_name=image_id, width=image.size[0], height=image.size[1], exp=expression,
                                    fid=image_id))
    '''finish load all data'''
    return cam_infos

def readNeRFBlendShapeDataset(path, eval, is_debug, novel_view, only_head):
    print("Load NeRFBlendShape Train Dataset")
    train_cam_infos = readNerfBlendShapeCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view,
                                                only_head=only_head)
    print("Load NeRFBlendShape Test Dataset")
    test_cam_infos = readNerfBlendShapeCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view,
                                               only_head=only_head)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info
