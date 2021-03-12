import glob
import os
from os.path import join
import shutil
from PIL import Image
import click


@click.command()
@click.option('--root-path')
@click.option('--output-path')
def main(root_path, output_path):
    len_frame = len(list(glob.glob(join(root_path, "semantic_labels", "*.png"))))

    os.makedirs(output_path, exist_ok=True)
    PER_SKIP = 5
    for i in range(0, len_frame):
        if i % PER_SKIP == 0:
            target_split = "val"
        else:
            target_split = "train"

        jpg_path = join(root_path, "JPEGImages", f"{i}.jpg")
        im = Image.open(jpg_path)
        image_root_path = join(output_path, "img_dir", target_split)
        os.makedirs(image_root_path, exist_ok=True)
        im.save(join(image_root_path, "{:06}.png".format(i)))

        semantic_root_path = join(output_path, "ann_dir", target_split)
        os.makedirs(semantic_root_path, exist_ok=True)
        semantic_label_path = join(semantic_root_path, "{:06}.png".format(i))
        shutil.copy(
            join(root_path, "semantic_labels", f"{i}.png"),
            semantic_label_path,
        )

if __name__ == "__main__":
    main()