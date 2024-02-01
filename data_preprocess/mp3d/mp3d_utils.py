"""
This code is an adaptation that uses Structured 3D for the code base.

Reference: https://github.com/bertjiazheng/Structured3D
"""

import numpy as np
from shapely.geometry import Polygon
import os
import json
import sys

sys.path.append('../.')
from common_utils import resort_corners


type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

mp3d2s3d = {'hallway': 'corridor', 'bathroom': 'bathroom', 'bedroom': 'bedroom', 'stairs': 'undefined', 'other room': 'undefined',
            'kitchen': 'kitchen', 'closet': 'store room', 'lounge': 'living room', 'living room': 'living room', 'balcony': 'balcony', 
            'dining room': 'dining room', 'entryway/foyer/lobby': 'corridor', 'office': 'office', 'porch/terrace/deck': 'garden', 
            'familyroom/lounge': 'living room', 'rec/game': 'studio', 'toilet': 'bathroom', 'spa/sauna': 'undefined', 'meetingroom/conferenceroom': 'office', 'powder room': 'undefined', 'elevator': 'undefined',
            'laundryroom/mudroom': 'laundry room', 'workout/gym/exercise': 'undefined', 'tv': 'studio', 'outdoor': 'undefined', 
            'utilityroom/toolroom': 'store room', 'junk': 'undefined', 'garage': 'garage', 'library': 'study', 'bar': 'undefined', 'wine closet': 'store room', 'other': 'undefined', 'storage': 'store room',
            'door': 'door', 'window': 'window'}



def generate_density(pc, width=256, height=256):

    # ps = point_cloud * -1
    # ps[:,0] *= -1
    # ps[:,1] *= -1

    image_res = np.array((width, height))

    max_coords = np.max(pc, axis=0)
    min_coords = np.min(pc, axis=0)
    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.05 * max_m_min
    min_coords = min_coords - 0.05 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res

    xy_range = (max_coords - min_coords)[:2]
    ratio = float(width) / max(xy_range)
    pc_res = np.array([int(x * ratio) for x in xy_range])
    normalization_dict["pc_res"] = pc_res

    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = np.round(
        (pc[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * pc_res[None] \
            + (image_res - pc_res) // 2
        )
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)), image_res - 1)
    # coordinates[:,1] = (image_res[1] - 1) - coordinates[:,1]

    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # print(np.unique(counts))
    # counts = np.minimum(counts, 1e2)

    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)


    return density, normalization_dict


def normalize_point(point, normalization_dict):
    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]
    pc_res = normalization_dict["pc_res"]
    point[:2] = [p * 1000. for p in point[:2]]
    point_2d = np.round(
        (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * pc_res \
            + (image_res - pc_res) // 2
        )
        
    point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)), image_res - 1)

    point[:2] = point_2d.tolist()

    # return point


def compute_hole_points2d(hole, ele):
    ele_p1, ele_p2 = np.array(ele['points'])[:, :-1]
    ele_len = np.linalg.norm(ele_p2 - ele_p1)

    disp_min, _ = hole['box']['min']
    disp_max, _ = hole['box']['max']
    hp1 = disp_min / ele_len * (ele_p2 - ele_p1) + ele_p1
    # hp_min[-1] += hh_min
    hp2 = disp_max / ele_len * (ele_p2 - ele_p1) + ele_p1
    # hp_max[-1] += hh_max
    hole_points = [
        [*hp1],
        [*hp2]
    ]
    # hole_points = np.array(hole_points)
    return hole_points


def normalize_annotations(arch_path, normalization_dict):
    arch_json = json.load(open(arch_path, "r"))
    
    for ele in arch_json["elements"]:
        if ele['type'] not in ['Floor', 'Wall']: continue
        
        if "holes" in ele and ele["holes"]:
            for hole in ele["holes"]:
                if hole['type'] not in ['Door', 'Window']: continue
                hole["points"] = compute_hole_points2d(hole, ele)
                for point in hole["points"]:
                    normalize_point(point, normalization_dict)
                    
        for point in ele["points"]:
            normalize_point(point, normalization_dict)
            
    for region in arch_json["regions"]:
        for point in region["points"]:
            normalize_point(point, normalization_dict)
    
    return arch_json


def parse_floor_plan_polys(annos, level):
    polygons = []
    for region in annos["regions"]:
        if region["level"] == level:
            polygons.append({'roomId': region['id'], 'type': region['type'], 'points': region['points']})
            
    for ele in annos["elements"]:
        ele_level = int(ele["id"].split('_')[0])
        if ele_level == level and "holes" in ele and ele["holes"]:
            for hole in ele["holes"]:
                if hole['type'] not in ['Door', 'Window']: continue
                # hole_points = compute_hole_points2d(hole, ele)
                polygons.append({'roomId': ele['roomId'], 'type': hole['type'].lower(), 'points': hole["points"]})

    return polygons


# def convert_lines_to_vertices(lines):
#     """
#     convert line representation to polygon vertices

#     """
#     polygons = []
#     lines = np.array(lines)

#     polygon = None
#     while len(lines) != 0:
#         if polygon is None:
#             polygon = lines[0].tolist()
#             lines = np.delete(lines, 0, 0)

#         lineID, juncID = np.where(lines == polygon[-1])
#         vertex = lines[lineID[0], 1 - juncID[0]]
#         lines = np.delete(lines, lineID, 0)

#         if vertex in polygon:
#             polygons.append(polygon)
#             polygon = None
#         else:
#             polygon.append(vertex)

#     return polygons



def generate_coco_dict(annos, polygons, curr_instance_id, curr_img_id, ignore_types):

    # junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    coco_annotation_dict_list = []

    # for poly_ind, (polygon, poly_type) in enumerate(polygons):
    for poly in polygons:
        if poly["type"] in ignore_types:
            continue

        polygon = np.asarray(poly["points"])[:, :2]
        
        if poly["type"] not in ['door', 'window']:
            poly_shapely = Polygon(polygon)
        else:
            poly_shapely = Polygon(np.append(polygon, polygon[0, None], axis=0))
        area = poly_shapely.area

        # assert area > 10
        # if area < 100:
        if poly["type"] not in ['door', 'window'] and area < 100:
            continue
        # if poly_type in ['door', 'window'] and area < 1:
        #     continue
        
        rectangle_shapely = poly_shapely.envelope

        # ### here we convert door/window annotation into a single line
        # if poly["type"] in ['door', 'window']:
        #     assert polygon.shape[0] == 4
        #     midp_1 = (polygon[0] + polygon[1])/2
        #     midp_2 = (polygon[1] + polygon[2])/2
        #     midp_3 = (polygon[2] + polygon[3])/2
        #     midp_4 = (polygon[3] + polygon[0])/2

        #     dist_1_3 = np.square(midp_1 -midp_3).sum()
        #     dist_2_4 = np.square(midp_2 -midp_4).sum()
        #     if dist_1_3 > dist_2_4:
        #         polygon = np.row_stack([midp_1, midp_3])
        #     else:
        #         polygon = np.row_stack([midp_2, midp_4])

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": curr_img_id,
                "bbox": coco_bb,
                "category_id": type2id[mp3d2s3d[poly["type"]]],
                "id": curr_instance_id}
        
        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1


    return coco_annotation_dict_list