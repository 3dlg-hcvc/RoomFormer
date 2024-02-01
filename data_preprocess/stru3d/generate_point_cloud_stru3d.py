import argparse
import os
from tqdm import tqdm
from PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_root', default='/datasets/external/Structured3D/data', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output_dir', default='/project/3dlg-hcvc/scan2arch/roomformer', type=str)
    args = a.parse_args()
    return args

def main(args):
    print("Creating point cloud from perspective views...")
    data_root = args.data_root

    # scenes = os.listdir(data_root)
    scenes = ["scene_00000"]
    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene)
        reader = PointCloudReaderPanorama(scene_path, random_level=0, generate_color=True, generate_normal=False)
        save_path = os.path.join(args.output_dir, 's3d', scene)
        os.makedirs(save_path, exist_ok=True)
        reader.export_ply(os.path.join(save_path, 'point_cloud.ply'))
            

if __name__ == "__main__":

    main(config())