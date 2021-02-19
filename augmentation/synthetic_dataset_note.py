#%%
import glob
import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np


from scipy.spatial.transform import Rotation as R

os.makedirs("/tmp/synthetic_images/", exist_ok=True)

class HandOverlayedImageGenerator:
    def __init__(self):
        self.cup = o3d.io.read_triangle_mesh("/home/yusuke/gitrepos/3d-tools/data/hands/hand1/scene.gltf", enable_post_processing=True)
        self.cup.compute_vertex_normals()
        self.cup.translate([0, 0, 0])
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.image_index = 0
        self.current_img = None

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)
        # vis.get_render_option().load_from_json("../../test_data/renderoption.json")
        vis.register_animation_callback(self.rotate_view)
        self.vis = vis

        self.vis.add_geometry(self.cup)
        # 
        # rotmat = R.from_rotvec([0, 2 * np.pi, 0]).as_matrix()
        # self.cup.rotate(rotmat)

    def _replace_image(self):
        if self.current_img is not None:
            self.vis.remove_geometry(self.current_img)
        bg_image = o3d.io.read_image(self.image_paths[self.image_index])
        self.vis.add_geometry(bg_image)
        self.current_img = bg_image

    def _save_image(self):
        image = self.vis.capture_screen_float_buffer(False)
        plt.imsave(os.path.join(self.to_dir, f"{self.image_index+1}.png"), np.asarray(image), dpi = 1)


    def rotate_view(self, vis):
        if self.image_index < len(self.image_paths):
            self._save_image()

            self._replace_image()
            # ctr = vis.get_view_control()
            self.cup.translate(np.random.random((3,)), relative=False)
            rotmat = R.random(random_state=1234).as_matrix()
            self.cup.rotate(rotmat)

        if self.image_index == len(self.image_paths):
            self._save_image()

        self.image_index += 1
        # self.cup.translate(np.random.random((3,)), relative=False)
        # if self.i % 4 == 0:
        #     self.cup.translate([1.0, 0, 0])
        # elif self.i % 4 == 1:
        #     self.cup.translate([0, 1.0, 0])
        # elif self.i % 4 == 2:
        #     self.cup.translate([-1.0, 0, 0])
        # elif self.i % 4 == 3:
        #     self.cup.translate([0, -1.0, 0])
        
        
        vis.update_geometry(self.cup)
        return False

    def generate(self, from_dir, to_dir):
        self.image_paths = list(glob.glob(os.path.join(from_dir, "JPEGImages/*.jpg")))
        self._replace_image()
        self.to_dir = to_dir
        os.makedirs(self.to_dir, exist_ok=True)
        self.vis.run()
        self.vis.destroy_window()

gen = HandOverlayedImageGenerator()
gen.generate("/home/yusuke/gitrepos/3d-tools/LINEMOD/onaho9", "/home/yusuke/gitrepos/3d-tools/LINEMOD/onaho9_occlusion")
# %%

