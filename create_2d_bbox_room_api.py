import argparse
import itertools
import json
import concurrent.futures
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Input
SUFFIX_RGB = '.rgb.png'  # RGB image. Used to visualize the bounding boxes
SUFFIX_SEGID = '.segments.png'
SUFFIX_INFO = '.info.json'
JSON_OBJ_ID_KEY = "mask_id"  # In json, this param contains the ID of the object in the mask
JSON_OBJ_LIST_KEY = "objects"  # In json, this param contains list of the object
# Output
SUFFIX_BBOX = '.bbox.json'
SUFFIX_VIZ = '.bbox.jpg'  # Viz of bboxes on RGB image

SEED_RANDOM = 0


def main(args):
    """Creates 2D Bounding Box for each object in an image from renders of Room API
    The info.json files in the Room API contain 3D bounding boxes for each object.
    Objects do not include the floor and walls. They can be shoes, belts, etc.

    We project the 3D bounding boxes to 2D bounding boxes in the image plane using the
    camera intrinsics and extrinsics and optionally visualize the boxes.
    """
    dir_src = Path(args.src_dir)
    if not dir_src.exists() or not dir_src.is_dir():
        raise ValueError(f"The given directory was not found: {dir_src}")
    if args.dst_dir:
        dir_dst = Path(args.dst_dir)
        if not dir_dst.exists():
            print(f"Creating output directory: {dir_dst}")
            dir_dst.mkdir(parents=True)
    else:
        dir_dst = dir_src

    list_infos = sorted(dir_src.glob('*' + SUFFIX_INFO))
    list_segids = sorted(dir_src.glob('*' + SUFFIX_SEGID))
    num_infos = len(list_infos)
    num_segids = len(list_segids)
    if num_infos < 1:
        raise ValueError(f"No {SUFFIX_INFO} files found in dir: {dir_src}")
    if num_segids < 1:
        raise ValueError(f"No {SUFFIX_SEGID} files found in dir: {dir_src}")
    if num_segids != num_infos:
        print(f"Error: The number of segmentindex files ({num_segids}) does not match"
              f" the number of info files ({num_infos}).\n    Calculating mismatching files...")

        img_nums_segid = [_img.name[:-len(SUFFIX_SEGID)] for _img in list_segids]
        img_nums_info = [_img.name[:-len(SUFFIX_INFO)] for _img in list_infos]
        raise ValueError(f"Mismatch in number of segmentindex and info files. These are the mismatching img numbers:\n"
                         f"{list(set(img_nums_segid) ^ set(img_nums_info))}")

    print(f'Creating bounding boxes from {num_infos} files.')

    if args.debug_viz_bbox_mask:
        create_viz = True
        print('Warning: debug visualizations enabled. This can be slow.')
        list_rgb = sorted(dir_src.glob('*' + SUFFIX_RGB))
        num_rgb = len(list_rgb)
        if num_segids != num_infos:
            print(f"Error: The number of RGB files ({num_rgb}) does not match"
                  f" the number of info files ({num_infos}).")

    else:
        create_viz = False
        list_rgb = itertools.repeat(None)

    if args.workers > 0:
        max_workers = args.workers
    else:
        max_workers = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(export_bbox_json, list_segids, list_infos, itertools.repeat(dir_dst), itertools.repeat(create_viz),
                     list_rgb)


def export_bbox_json(file_segid, file_info_json, dst_dir, create_viz=False, file_rgb=None):
    """Export a json file containing info about bounding box of each object in a container from a given
    segmentindex and info.json file
    The segmentindex and info.json file should have the same image number. The exported json with bounding box info
    will also have the same image number.

    Args:
        file_segid (str or pathlib.Path): Path to png file containing masks of all objects
        file_info_json (str or pathlib.Path): Path to a info.json file containing info camera intrinsics+extrinsics
                                              as well as of all objects in the scene.
        dst_dir (str or pathlib.Path): The directory where json files with the 2D bboxes will be exported
        create_viz (Optional, bool): If given, will create a visualization of the mask and bounding box of
                                     EVERY valid object in EVERY image. Warning: This is very slow.
        file_rgb (Optional, str or pathlib.Path): If create_viz is true, the RGB file is used to visualize the
                                                  bounding boxes.
    """
    # Check inputs
    file_segid = Path(file_segid)
    file_info_json = Path(file_info_json)
    dir_dst = Path(dst_dir)
    img_num_segid = file_segid.name[:-len(SUFFIX_SEGID)]
    img_num_info = file_info_json.name[:-len(SUFFIX_INFO)]
    if img_num_segid != img_num_info:
        raise ValueError(f"The image number of segid file ({img_num_segid}) and info file ({img_num_info})"
                         f"are different: {file_segid} {file_info_json}")
    if not dir_dst.exists():
        print(f"Creating output dir: {dir_dst}")
        dir_dst.mkdir(parents=True)

    # Read input data
    segid = cv2.imread(str(file_segid), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    with open(file_info_json) as json_file:
        info = json.load(json_file)
    if create_viz:
        if file_rgb is None:
            raise ValueError('No RGB image given. For visuzalizing bboxes, RGB image should be passed.')

        img_rgb = cv2.imread(str(file_rgb))

    new_obj_dict = {}  # Stores info that will be output
    new_obj_dict[JSON_OBJ_LIST_KEY] = []
    random.seed(SEED_RANDOM)
    # Process all objects in the scene
    for obj in info[JSON_OBJ_LIST_KEY]:
        # Get ID of the object in the mask
        obj_id = obj[JSON_OBJ_ID_KEY]

        # Get mask of the object
        mask_obj = (segid == obj_id).astype(np.uint8)  # Convert bool to int

        # Get bounding box from mask
        maskx = np.any(mask_obj, axis=0)
        masky = np.any(mask_obj, axis=1)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1])
        bbox_obj = {"bounding-box-2d": {"x_min": int(x1), "x_max": int(x2), "y_min": int(y1), "y_max": int(y2)}}
        obj.update(bbox_obj)
        new_obj_dict[JSON_OBJ_LIST_KEY].append(obj)

        # Debug Visualizations - Export an image of mask and bounding box for every object in an image
        if create_viz:
            # Draw bounding box
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), rand_color, 2)

    filename = dir_dst.joinpath(img_num_info + SUFFIX_BBOX)
    print(f"Saving bbox json: {filename}")
    with open(filename, 'w') as outfile:
        json.dump(new_obj_dict, outfile, indent=4)

    if create_viz:
        filename = dir_dst.joinpath(img_num_info + SUFFIX_VIZ)
        print(f"Saving bbox viz: {filename}")
        cv2.imwrite(str(filename), img_rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 2D Bounding Box for each object in an image from renders"
                                                 " of Room API")
    parser.add_argument("-s", "--src_dir", help="Dir where container segmentindex.png and info.json files are stored",
                        required=True)
    parser.add_argument("-d", "--dst_dir", help="Optional. Dir where json files of bounding boxes should be saved",
                        required=False, default=None)
    parser.add_argument("--debug_viz_bbox_mask", help="Visualize the mask and bounding box of EVERY object in EVERY "
                                                      "image. Warning: This is very slow.", required=False,
                        action="store_true")
    parser.add_argument("-w", "--workers", help="Number of processes to use. "
                                                "Defaults to the number of processors on the machine.", default=0,
                        type=int)
    args = parser.parse_args()
    main(args)
