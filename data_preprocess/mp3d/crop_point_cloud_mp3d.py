import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm


data_dir = "/datasets/internal/matterport/annotated_house_mesh"
arch_dir = "/project/3dlg-hcvc/rlsd/data/mp3d/arch_refined_clean"
output_dir = "/project/3dlg-hcvc/scan2arch/roomformer/mp3d"

house_ids = [f.split('.')[0] for f in os.listdir(arch_dir)]

# houses = {}
for house_id in tqdm(house_ids):
    # house = houses.setdefault(house_id, {})
    arch = json.load(open(os.path.join(arch_dir, f"{house_id}.arch.json")))
    regions = {}
    for region in arch["regions"]:
        level = int(region["level"])
        region_heights = regions.setdefault(level, [])
        region_z = np.asarray(region["points"])[:, -1].min()
        region_heights.append(region_z)
        
    level_heights = []
    for level in range(len(regions)):
        level_height = np.array(regions[level]).min()
        level_heights.append(level_height)
        # house[level] = level_height
    level_heights.append(10000.)
    level_ends = list(zip(level_heights[:-1], level_heights[1:]))

    pcd = o3d.io.read_point_cloud(os.path.join(data_dir, house_id, f"{house_id}.faceids.ply"))
    all_points = np.asarray(pcd.points)
    all_colors = np.asarray(pcd.colors)
    # print(f"# of points: {len(points)}")
    
    for i, (level_lo, level_hi) in enumerate(tqdm(level_ends, leave=False)):
        valid = (all_points[:, -1] >= level_lo) & (all_points[:, -1] < level_hi)
        points = all_points[valid] * 1000.
        colors = all_colors[valid]
        # print(f"# of crop points: {len(points)}")

        points[:,:2] = np.round(points[:,:2] / 10) * 10.
        points[:,2] = np.round(points[:,2] / 100) * 100.
        unique_coords, unique_ind = np.unique(points, return_index=True, axis=0)
        points = points[unique_ind]
        colors = colors[unique_ind]

        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(points)
        crop_pcd.colors = o3d.utility.Vector3dVector(colors)

        os.makedirs(os.path.join(output_dir, f"{house_id}_L{i}"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{house_id}_L{i}", 'point_cloud.crop.ply'), crop_pcd)