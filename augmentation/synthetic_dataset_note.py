#%%
import open3d as o3d
import open3d.visualization.rendering as rendering

render = rendering.OffscreenRenderer(1280, 960)

yellow = rendering.Material()
yellow.base_color = [1.0, 0.75, 0.0, 1.0]
yellow.shader = "defaultLit"

green = rendering.Material()
green.base_color = [0.0, 0.5, 0.0, 1.0]
green.shader = "defaultLit"

grey = rendering.Material()
grey.base_color = [0.7, 0.7, 0.7, 1.0]
grey.shader = "defaultLit"

white = rendering.Material()
white.base_color = [1.0, 1.0, 1.0, 1.0]
white.shader = "defaultLit"

cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
cyl.compute_vertex_normals()
cyl.translate([-2, 0, 1.5])
sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
sphere.compute_vertex_normals()
sphere.translate([-2, 0, 3])

box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
box.compute_vertex_normals()
box.translate([-1, -1, 0])
solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
solid.compute_triangle_normals()
solid.compute_vertex_normals()
solid.translate([0, 0, 1.75])

texture = rendering.Material()
texture.shader = "defaultLit"


# cup.translate([0, -10, 0])
# cup.textures.append(o3d.io.read_image("/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/keyboard_2/textures/Keyboard_Body1_baseColor.png"))
# cup.textures.append(o3d.io.read_image("/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/keyboard_2/textures/Keyboard_Keys_baseColor.png"))
# texture.base_color = [0,0,0,0]
# texture.generic_imgs['aaa'] = o3d.io.read_image("/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/keyboard_2/textures/Keyboard_Keys_baseColor.png")
# texture.normal_img = o3d.io.read_image("/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/keyboard_2/textures/Keyboard_Keys_normal.png")
# texture.roughness_img = o3d.io.read_image("/home/yusuke/gitrepos/ai-waifu-experiments/datapipeline_for_onaho/keyboard_2/textures/Keyboard_Keys_metallicRoughness.png")
import matplotlib.pyplot as plt
import numpy as np

# def get_open_box_mesh():
#     mesh = o3d.geometry.TriangleMesh.create_box()
#     mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:-2])
#     mesh.compute_vertex_normals()
#     mesh.rotate(
#         mesh.get_rotation_matrix_from_xyz((0.8 * np.pi, 0, 0.66 * np.pi)),
#         center=mesh.get_center(),
#     )
#     return mesh

# print(image.())

# def visualize_and_save(geometries):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     for geometry in geometries:
#         print(vis.add_geometry(geometry))
#     vis.get_render_option().load_from_json("../../test_data/renderoption.json")
#     vis.run()
#     image = vis.capture_screen_float_buffer(False)
#     plt.imsave("/tmp/test3.png", np.asarray(image), dpi = 1)
#     vis.destroy_window()

from scipy.spatial.transform import Rotation as R

os.makedirs("/tmp/synthetic_images/", exist_ok=True)

class SynGen:
    def __init__(self):
        self.cup = o3d.io.read_triangle_mesh("/home/yusuke/gitrepos/3d-tools/data/hand/scene.gltf", enable_post_processing=True)
        self.cup.compute_vertex_normals()
        self.cup.translate([0, 0, 10])
        self.i = 0
    def rotate_view(self, vis):
        rotmat = R.random(random_state=1234).as_matrix()
        self.i += 1
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(f"/tmp/synthetic_images/{self.i}.png", np.asarray(image), dpi = 1)
        # ctr = vis.get_view_control()
        self.cup.rotate(rotmat)
        if self.i % 4 == 0:
            self.cup.translate([1.0, 0, 0])
        elif self.i % 4 == 1:
            self.cup.translate([0, 1.0, 0])
        elif self.i % 4 == 2:
            self.cup.translate([-1.0, 0, 0])
        elif self.i % 4 == 3:
            self.cup.translate([0, -1.0, 0])
        
        
        vis.update_geometry(self.cup)
        return False

    def generate(self):
        image_path = "/home/yusuke/gitrepos/3d-tools/LINEMOD/onaho9/JPEGImages/758.jpg"
        image = o3d.io.read_image(image_path)
        o3d.visualization.draw_geometries_with_animation_callback([self.cup, image],
                                                        gen.rotate_view, width=640, height=480)


gen = SynGen()
gen.generate()
# visualize_and_save([cup, image])
# visualize_and_save([image])

print(cup.textures)
print(cup.has_textures())

# render.scene.add_geometry("cup", cup, texture)
# # render.scene.add_geometry("cyl", cyl, green)
# render.scene.add_geometry("sphere", sphere, yellow)
# render.scene.add_geometry("box", box, grey)
# render.scene.add_geometry("solid", solid, white)
# render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
# render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
#                                     75000)
# # render.scene.scene.add_directional_light("dir_light", (1, 0, 0), )
# # render.scene.scene.add_spot_light("spot_light", (1.0, 1, 1), [0,0,0], [0, 10, 0], 75000.0, 0, 0, 0, True)
# render.scene.scene.enable_sun_light(True)
# # render.scene.show_axes(True)

# img = render.render_to_image()
# # o3d.io.write_image("/tmp/test.png", img, 9)

# render.scene.camera.look_at([0, 0, 0], [-10, 0, 0], [0, 0, 1])
# img = render.render_to_image()
# # o3d.io.write_image("/tmp/test2.png", img, 9)


# %%
