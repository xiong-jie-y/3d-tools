import glob
import json
import os
import shutil

output_dirs = [
    "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho9",
    "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho6",
    "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho_open3d_1",
    "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho_open3d_2",
    # "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho_open3d_3",
]

# output_dirs = [
#     "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho9",
#     "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho9_occlusion1",
# ]

destination_root = "/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/LINEMOD/onaho_coco"

if os.path.exists(destination_root):
    shutil.rmtree(destination_root)

os.makedirs(os.path.join(destination_root, "JPEGImages"))

class AnnotationMerger:
    def __init__(self):
        self.new_annotation_id = 0
        self.image_id_mapper = {}
        self.image_id = 0

        self.annotations = []
        self.images = []
        self.categories = []

    def add_annotations(self, annotation_path):
        annotations_json = json.load(open(annotation_path))
        
        for image in annotations_json["images"]:
            image["file_name"] = "{}_{}".format(dataset_id, image["file_name"])
            self.image_id_mapper[image["id"]] = self.image_id
            image["id"] = self.image_id
            image["image_id"] = self.image_id

            self.image_id += 1
            self.images.append(image)

        for annotation in annotations_json["annotations"]:
            annotation["image_id"] = self.image_id_mapper[annotation["image_id"]]
            annotation["id"] = self.new_annotation_id

            self.new_annotation_id += 1
            self.annotations.append(annotation)

        self.categories = annotations_json['categories']

    def dump(self, path):
        json.dump({
            "annotations": self.annotations,
            "images": self.images,
            "categories": self.categories
        }, open(path, "w"))
        

annotation_merger = AnnotationMerger()
annotation_val_merger = AnnotationMerger()

for dataset_id, output_dir in enumerate(output_dirs):
    len_frame = len(list(glob.glob(os.path.join(output_dir, "JPEGImages/*.jpg"))))

    for i in range(0, len_frame):
        shutil.copy(
            os.path.join(output_dir, f"JPEGImages/{i}.jpg"),
            os.path.join(destination_root, f"JPEGImages/{dataset_id}_{i}.jpg")
        )

    annotation_merger.add_annotations(os.path.join(output_dir, "annotations.json"))
    annotation_val_merger.add_annotations(os.path.join(output_dir, "annotations_val.json"))

# import IPython; IPython.embed()

annotation_merger.dump(os.path.join(destination_root, "annotations.json"))
annotation_val_merger.dump(os.path.join(destination_root, "annotations_val.json"))