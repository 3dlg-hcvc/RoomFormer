import os
import argparse
from tqdm import tqdm
from PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_dir', default='/project/3dlg-hcvc/rlsd/data/mp3d', type=str, help='path to raw mp3d_panorama folder')
    a.add_argument('--house_level_id', default='', type=str)
    a.add_argument('--output_dir', default='/project/3dlg-hcvc/scan2arch/roomformer/mp3d', type=str)
    args = a.parse_args()
    return args


def main(args):
    print("Creating point cloud from panoramas...")
    data_dir = args.data_dir
    
    # full_pano_ids = [p.strip() for p in open("/project/3dlg-hcvc/rlsd/data/mp3d/pano_ids.txt")]
    full_pano_ids = ["5ZKStnWn8Zo_L1_34557b20f94f4e70b7604ab749d44d61"]
    house2panos = {}
    for id in full_pano_ids:
        house_id, level_id, pano_id = id.split("_")
        if not (args.house_level_id and args.house_level_id == f'{house_id}_{level_id}'):
            continue
        house = house2panos.setdefault(house_id, {})
        level = house.setdefault(level_id, [])
        level.append(pano_id)
        
    for house_id in tqdm(house2panos):
        house = house2panos[house_id]
        for level_id in house:
            panos = house[level_id]
            reader = PointCloudReaderPanorama(panos, house_id, data_dir, random_level=0, generate_color=True, generate_normal=False)
            save_path = os.path.join(args.output_dir, f'{house_id}_{level_id}')
            os.makedirs(save_path, exist_ok=True)
            reader.export_ply(os.path.join(save_path, 'point_cloud.arch.ply'))

            

if __name__ == "__main__":

    main(config())