import os
import argparse
import numpy as np
import BboxToolkit
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from pycocotools.coco import COCO


def generate_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-coco-file", help="path to input a coco annotation file (*.json)")
    parser.add_argument("-o", "--output-dir", help="path to output Dota format annotation files (*.txt)")
    return parser.parse_args()


@dataclass
class OrientedBBox:
    category_name: str
    category_id: int
    x: float
    y: float
    w: float
    h: float
    theta: float  # [rad]

    def __str__(self):
        return f"x:{self.x}, y:{self.y}, w:{self.w}, h:{self.h}, theta:{self.theta * 180 / np.pi}[deg]"

    @property
    def xywht(self):
        return self.x, self.y, self.w, self.h, self.theta

    @property
    def xywht_as_str(self):
        return f"{self.x}, {self.y}, {self.w}, {self.h}, {self.theta}"

    @property
    def polypoints_as_str(self):
        polypoints = BboxToolkit.obb2poly(self.xywht)
        poly_str = f"{polypoints[0]}"
        for value in polypoints[1:]:
            poly_str += f" {value}"
        return poly_str


def main():
    args = generate_argument_parser()

    """ Load dataset
    """
    coco_data = COCO(args.input_coco_file)
    image_file_dict_id2name = {image_data["id"]: image_data["file_name"] for image_data in coco_data.imgs.values()}
    category_dict_id2label = {category_data["id"]: category_data["name"] for category_data in coco_data.cats.values()}

    """ Assign empty list to corresponding image
    """
    dict_image_name_to_obbs: Dict[str, List[OrientedBBox]] = {}
    for image_file_name in image_file_dict_id2name.values():
        dict_image_name_to_obbs[image_file_name] = []

    for annotation_data in coco_data.anns.values():
        """Load image and annotation data"""
        image_id = annotation_data["image_id"]
        image_name = image_file_dict_id2name[image_id]
        category_id = annotation_data["category_id"]

        """ Load and convert oriented bounding-box information
        """
        assert len(annotation_data["segmentation"]) == 1

        obb_polygon = annotation_data["segmentation"][0]  # obb: oriented bbox, quadruple pairs of (x_i, y_i)
        obb_ann_info = BboxToolkit.poly2obb(np.asarray(obb_polygon))
        obb = OrientedBBox(
            category_name=category_dict_id2label[category_id],
            category_id=category_id,
            x=obb_ann_info[0],
            y=obb_ann_info[1],
            w=obb_ann_info[2],
            h=obb_ann_info[3],
            theta=obb_ann_info[4],
        )
        dict_image_name_to_obbs[image_name].append(obb)

    """ Output data
    """
    output_dir_pathlib = Path(args.output_dir)

    """ Assign empty annotation text to corresponding key
    """
    dict_image_name_to_annotation_text = {}
    for image_name in dict_image_name_to_obbs.keys():
        dict_image_name_to_annotation_text[image_name] = ""

    """ Generate annotation text data to output
    """
    for image_name, obb_list in dict_image_name_to_obbs.items():
        for obb in obb_list:
            annotation_str = f"{obb.polypoints_as_str} {obb.category_name} 0"
            dict_image_name_to_annotation_text[image_name] += annotation_str

    """ Dump
    """
    for image_name, annotation_text_data in dict_image_name_to_annotation_text.items():
        image_name_stem = output_dir_pathlib.joinpath(image_name).stem
        output_text_file_pathlib = Path(args.output_dir, f"{image_name_stem}.txt")
        if output_text_file_pathlib.exists():
            os.remove(str(output_text_file_pathlib))
        with open(str(output_text_file_pathlib), "w") as f:
            f.write(annotation_text_data)

    print("done")


if __name__ == "__main__":
    main()
