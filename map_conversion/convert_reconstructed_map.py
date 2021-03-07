import os
import glob
from os.path import join
import shutil
import click
import numpy as np
import json

from numpy.lib.utils import source

class SingleShotPoseGenerator:
    ":Broken"
    def output_rgb_files(self, src_rgb_paths):
        # Create images.
        jpeg_images_path = os.path.join(destination_root_path, "JPEGImages")
        os.makedirs(jpeg_images_path, exist_ok=True)
        for src_rgb_path in src_rgb_paths:
            shutil.copy(
                src_rgb_path,
                os.path.join(jpeg_images_path, f"{i}.jpg")
            )

    def output_depth_files(self, src_depth_paths):
        depths_path = os.path.join(destination_root_path, "depth")
        os.makedirs(depths_path, exist_ok=True)
        for src_depth_path in src_depth_paths:
            shutil.copy(
                src_depth_path,
                os.path.join(depths_path, f"{i}.png")
            )

    def output_trajectory(self, trajectory):
        np.save(open(os.path.join(destination_root_path, "transforms.npy"), "wb"), np.array(mats))

    def output_intrinsic(self,intrinsic):
        json.dump({
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "fx": intrinsics["intrinsic_matrix"][0],
        "fy": intrinsics["intrinsic_matrix"][4],
        "ppx": intrinsics["intrinsic_matrix"][6],
        "ppy": intrinsics["intrinsic_matrix"][7],
        "depth_scale": 0.0010000000474974513
      }, open(os.path.join(destination_root_path, "intrinsics.json"), "w"))

class Intrinsic3DDatasetGenerator:
    pass

class DscastarGenerator: 
    def __init__(self, output_root):
        self.output_root = join(output_root, "train")
        self.seq = 0
        self.frame_length = None

    def output_rgb_files(self, src_rgb_paths):
        self.frame_length = len(src_rgb_paths)
        rgb_folder = join(self.output_root, 'rgb')
        os.makedirs(rgb_folder, exist_ok=True)
        for i, src_rgb_path in enumerate(src_rgb_paths):
            shutil.copy(
                src_rgb_path,
                join(rgb_folder, "{}-frame-{:06}.color.jpg".format(self.seq, i))
            )
    
    def output_depth_files(self, src_depth_paths):
        depth_folder = join(self.output_root, 'depth')
        os.makedirs(depth_folder, exist_ok=True)
        for i, src_depth_path in enumerate(src_depth_paths):
            shutil.copy(
                src_depth_path,
                join(depth_folder, "{}-frame-{:06}.depth.png".format(self.seq, i))
            )

    def output_trajectory(self, pose_matrixes):
        poses_folder = join(self.output_root, 'poses')
        os.makedirs(poses_folder, exist_ok=True)
        for i, pose_matrix in enumerate(pose_matrixes):
            lines = []
            for row in pose_matrix:
                # This need to end with \n.
                # Line doesn't mean something that end with \n.
                lines.append(" ".join(row.astype(np.str)) + "\n")
            open(join(poses_folder, "{}-frame-{:06}.pose.txt".format(self.seq, i)), "w").writelines(lines)

    def output_intrinsic(self, intrinsic):
        calibration_folder = join(self.output_root, 'calibration')
        os.makedirs(calibration_folder, exist_ok=True)
        assert (self.frame_length)
        for i in range(0, self.frame_length):
            open(join(calibration_folder, "{}-frame-{:06}.calibration.txt".format(self.seq, i)), "w") \
                .write(str(intrinsic["intrinsic_matrix"][0]))

@click.command()
@click.option("--source-path")
@click.option("--destination-path")
@click.option("--to-type")
def main(source_path, destination_path, to_type):
    source_path = os.path.expanduser(source_path)
    destination_root_path = os.path.expanduser(destination_path)
    generator = None
    if to_type == "dsacstar":
        generator = DscastarGenerator(destination_root_path)
    else:
        raise RuntimeError(f"no to_type = {to_type}.")

    os.makedirs(destination_root_path, exist_ok=True)

    len_frame = len(list(glob.glob(os.path.join(source_path, "color/*.jpg"))))

    # Convert rgbs.
    open3d_rgb_paths = []
    for i in range(0, len_frame):    
        open3d_rgb_paths.append(os.path.join(source_path, "color", '{:06}.jpg'.format(i)))
    generator.output_rgb_files(open3d_rgb_paths)

    # Convert depths.
    open3d_depth_paths = []
    for i in range(0, len_frame):
        open3d_depth_paths.append(os.path.join(source_path, "depth", '{:06}.png'.format(i)))
    generator.output_depth_files(open3d_depth_paths)

    # Convert Trajectory.
    lines = open(os.path.join(source_path, "scene/trajectory.log"), 'r').readlines()
    mats = []
    for i in range(0, len_frame * 5, 5):
        rows = [
            [float(t) for t in lines[i + 1].split(" ")],
            [float(t) for t in lines[i + 2].split(" ")],
            [float(t) for t in lines[i + 3].split(" ")],
            [float(t) for t in lines[i + 4].split(" ")]
        ]
        print(rows)
        # mats.append(np.linalg.inv(np.array(rows)))
        mats.append(np.array(rows))
    generator.output_trajectory(mats)    

    # Convert intrinsic.
    intrinsics = json.load(open(os.path.join(source_path, "camera_intrinsic.json")))
    generator.output_intrinsic(intrinsics)

if __name__ == "__main__":
    main()