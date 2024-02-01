import argparse
import json
import os
import sys
from tqdm import tqdm
from mp3d_utils import generate_density, normalize_annotations, parse_floor_plan_polys, generate_coco_dict

sys.path.append('../.')
from common_utils import read_scene_pc, export_density


type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

mp3d2s3d = {'hallway': 'corridor', 'bathroom': 'bathroom', 'bedroom': 'bedroom', 'stairs': 'undefined', 'other room': 'undefined',
            'kitchen': 'kitchen', 'closet': 'store room', 'lounge': 'living room', 'balcony': 'balcony', 'dining room': 'dining room',
            'entryway/foyer/lobby': 'corridor', 'office': 'office', 'porch/terrace/deck': 'garden', 'familyroom/lounge': 'living room',
            'rec/game': 'studio', 'toilet': 'bathroom', 'spa/sauna': 'undefined', 'meetingroom/conferenceroom': 'office', 
            'laundryroom/mudroom': 'laundry room', 'workout/gym/exercise': 'undefined', 'tv': 'studio', 'outdoor': 'undefined', 
            'utilityroom/toolroom': 'store room', 'junk': 'undefined', 'garage': 'garage', 'library': 'study',
            'door': 'door', 'window': 'window'}

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for mp3d point cloud')
    a.add_argument('--data_dir', default='/project/3dlg-hcvc/scan2arch/roomformer/mp3d', type=str, help='path to mp3d point cloud folder')
    a.add_argument('--output_dir', default='/localhome/qiruiw/research/RoomFormer/data/mp3d', type=str, help='path to output folder')
    a.add_argument('--arch_dir', default='/project/3dlg-hcvc/rlsd/data/mp3d/arch_refined_clean', type=str, help='path to output folder')

    args = a.parse_args()
    return args

def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    arch_dir = args.arch_dir

    # ### prepare
    annotation_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotation_dir, exist_ok=True)

    # train_img_folder = os.path.join(output_dir, 'train')
    # os.makedirs(train_img_folder, exist_ok=True)
    # val_img_folder = os.path.join(output_dir, 'val')
    # os.makedirs(val_img_folder, exist_ok=True)
    test_img_folder = os.path.join(output_dir, 'test')
    os.makedirs(test_img_folder, exist_ok=True)

    # coco_train_json_path = os.path.join(annotation_dir, 'train.json')
    # coco_val_json_path = os.path.join(annotation_dir, 'val.json')
    coco_test_json_path = os.path.join(annotation_dir, 'test.json')

    # coco_train_dict = {"images":[],"annotations":[],"categories":[]}
    # coco_val_dict = {"images":[],"annotations":[],"categories":[]}
    coco_test_dict = {"images":[],"annotations":[],"categories":[]}

    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        # coco_train_dict["categories"].append(type_dict)
        # coco_val_dict["categories"].append(type_dict)
        coco_test_dict["categories"].append(type_dict)

    ### begin processing
    instance_id = 0
    scenes = sorted(os.listdir(data_dir))
    for img_id, scene_id in enumerate(tqdm(scenes)):
        # scene_path = os.path.join(data_root, 'Structured3D', scene)
        # scene_id = scene.split('_')[-1]
        house_id, level_id = scene_id.split('_')
        level = int(level_id[1:])
        arch_path = os.path.join(arch_dir, f'{house_id}.arch.json')

        # if int(scene_id) in invalid_scenes_ids:
        #     print('skip {}'.format(scene))
        #     continue
        
        # load pre-generated point cloud 
        ply_path = os.path.join(data_dir, scene_id, 'point_cloud.crop.ply')
        points = read_scene_pc(ply_path)
        xyz = points[:, :3]

        ### project point cloud to density map
        density, normalization_dict = generate_density(xyz, width=256, height=256)
        
        ### rescale raw annotations
        normalized_annos = normalize_annotations(arch_path, normalization_dict)

        ### prepare coco dict
        # img_id = int(scene_id)
        img_dict = {}
        img_dict["file_name"] = scene_id + '.png'
        img_dict["id"] = img_id
        img_dict["width"] = 256
        img_dict["height"] = 256

        ### parse annotations
        polys = parse_floor_plan_polys(normalized_annos, level)
        polygons_list = generate_coco_dict(normalized_annos, polys, instance_id, img_id, ignore_types=['outwall'])

        instance_id += len(polygons_list)

        # ### train
        # if int(scene_id) < 3000:
        #     coco_train_dict["images"].append(img_dict)
        #     coco_train_dict["annotations"] += polygons_list
        #     export_density(density, train_img_folder, scene_id)

        # ### val
        # elif int(scene_id) >= 3000 and int(scene_id) < 3250:
        #     coco_val_dict["images"].append(img_dict)
        #     coco_val_dict["annotations"] += polygons_list
        #     export_density(density, val_img_folder, scene_id)

        ### test
        # else:
        coco_test_dict["images"].append(img_dict)
        coco_test_dict["annotations"] += polygons_list
        export_density(density, test_img_folder, scene_id)
        
        # export_density(density, os.path.join(data_dir, scene_id), scene_id)
        print(scene_id)


    # with open(coco_train_json_path, 'w') as f:
    #     json.dump(coco_train_dict, f)
    # with open(coco_val_json_path, 'w') as f:
    #     json.dump(coco_val_dict, f)
    with open(coco_test_json_path, 'w') as f:
        json.dump(coco_test_dict, f)


if __name__ == "__main__":

    main(config())