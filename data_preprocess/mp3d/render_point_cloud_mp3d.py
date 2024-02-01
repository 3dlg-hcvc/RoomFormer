import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import trimesh
import pyrender
from pyrender import RenderFlags
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer


def render_topdown_view(in_file, out_file):
    # output_folder = os.path.join(args.output, *args.object.split('/')[-2:])
    # os.makedirs(output_folder, exist_ok=True)
    # if args.skip_done and len(glob(os.path.join(output_folder, f"render-*.png"))) == args.renders:
    #     return

    pc = trimesh.load(in_file)
    pc.vertices /= 1000.
    pc.vertices[:, 1] *= -1.
    valid = pc.vertices[:, -1] < pc.vertices[:, -1].max() - 0.5
    pc.vertices = pc.vertices[valid]
    pc.colors = pc.colors[valid]

    center = np.mean(pc.bounds, axis=0)
    radius = np.linalg.norm(pc.vertices[:, :2] - center[:2], axis=1).max()
    # scale = 1.0 / float(max(obj.bounds[1] - obj.bounds[0]))
    center_mat = np.array([
    [1, 0, 0, -center[0]],
    [0.0, 1, 0.0, -center[1]],
    [0.0, 0.0, 1, -center[2]],
    [0.0,  0.0, 0.0, 1.0]
    ])
    # norm_mat = np.array([
    #     [scale, 0, 0, 0],
    #     [0.0, scale, 0.0, 0],
    #     [0.0, 0.0, scale, 0],
    #     [0.0,  0.0, 0.0, 1.0]
    # ])
    # obj.apply_transform(np.matmul(norm_mat, center_mat))
    pc.apply_transform(center_mat)
    
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
    pc_mesh = pyrender.Mesh.from_points(points=pc.vertices, colors=pc.colors)
    scene.add(pc_mesh)

    direc_l = DirectionalLight(color=np.ones(3), intensity=1)
    # spot_l = SpotLight(color=np.ones(3), intensity=1.0, innerConeAngle=np.pi/8, outerConeAngle=np.pi/3)
    # point_l = PointLight(color=np.ones(3), intensity=10.0)
    r = OffscreenRenderer(viewport_width=500, viewport_height=500)

    # randomize camera settings
    dis = 30
    fov = np.arctan2(radius+1, dis) * 2
    # azim = 0 # - np.pi / 5 #
    # elev = - np.pi / 6 #
    cam_pose = np.eye(4)
    cam_pose[2, 3] = dis
    # rotx = np.array([
    #     [1.0, 0, 0],
    #     [0.0, np.cos(elev), np.sin(elev)],
    #     [0.0, -np.sin(elev), np.cos(elev)]
    # ])
    # roty = np.array([
    #     [np.cos(azim), 0, np.sin(azim)],
    #     [0.0, 1, 0.0],
    #     [-np.sin(azim), 0, np.cos(azim)]
    # ])
    # rotz = np.array([
    #     [np.cos(azim), -np.sin(azim), 0.],
    #     [np.sin(azim), np.cos(azim), 0.0],
    #     [0., 0., 1.]
    # ])
    # cam_pose[:3, :3] = np.matmul(rotz, rotx)
    cam = PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    cam_node = scene.add(cam, pose=cam_pose)
    direc_l_node = scene.add(direc_l)
    # direc_l_node = scene.add(direc_l, pose=cam_pose)
    
    color, _ = r.render(scene)
    Image.fromarray(color).save(out_file)
    
    scene.remove_node(cam_node)
    # scene.remove_node(direc_l_node)

    r.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip rendering')
    parser.add_argument('--skip_done', default=False, action='store_true',
                        help='Skip objects exist in output folder')
    args = parser.parse_args()
    
    scenes = sorted(os.listdir("/project/3dlg-hcvc/scan2arch/roomformer/mp3d"))
    for scene in tqdm(scenes):
        render_topdown_view(f"/project/3dlg-hcvc/scan2arch/roomformer/mp3d/{scene}/point_cloud.crop.ply",
                            f"/project/3dlg-hcvc/scan2arch/roomformer/mp3d/{scene}/topdown.png")
    
    