# https://github.com/CapObvios/Depth-Map-Visualizer/blob/master/DepthToObj.py
import argparse
import numpy as np
import cv2
import math
import os
from jhutil import inverse_3x4
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--depth_path', dest='depth_path',
                        help='depth map path',
                        default='objaverse-rendering/example/example_obj/000_depth.npy', type=str)
    parser.add_argument('--depth_invert', dest='depth_invert',
                        help='Invert depth map',
                        default=False, action='store_true')
    parser.add_argument('--rgb_path', dest='rgb_path',
                        help='corresponding image path',
                        default='objaverse-rendering/example/example_obj/000.png', type=str)
    parser.add_argument('--obj_path', dest='obj_path',
                        help='output path of .obj file',
                        default='model.obj', type=str)
    parser.add_argument('--mtl_path', dest='mtl_path',
                        help='output path of .mtl file',
                        default='model.mtl', type=str)
    parser.add_argument('--mat_name', dest='mat_name',
                        help='name of material to create',
                        default='colored', type=str)
    parser.add_argument('--extrinsic_path', dest='extrinsic_path',
                        help='path to extrinsic matrix',
                        default='objaverse-rendering/example/example_obj/000.npy', type=str)
    parser.add_argument('--fov', dest='fov',
                        help='field of view',
                        default=0.8575560548920328, type=float)
    args = parser.parse_args()
    return args

def merge_image(rgb_path, back_rgb_path, merged_rgb_path):
    # Open the images
    image1 = Image.open(rgb_path)
    image2 = Image.open(back_rgb_path)

    # Ensure the images are the same size
    assert image1.size == image2.size, "Images are not the same size"

    # Create a new image with a width that is the sum of both image widths and the same height
    merged_image = Image.new('RGB', (image1.width + image2.width, image1.height))

    # Paste the images side by side
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (image1.width, 0))

    # Save the merged image
    merged_image.save(merged_rgb_path)


def create_mtl(mtl_path, mat_name, rgb_path):
    if max(mtl_path.find('\\'), mtl_path.find('/')) > -1:
        os.makedirs(os.path.dirname(mtl_path), exist_ok=True)
    with open(mtl_path, "w") as f:
        f.write("newmtl " + mat_name + "\n")
        f.write("Ns 10.0000\n")
        f.write("d 1.0000\n")
        f.write("Tr 0.0000\n")
        f.write("illum 2\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("map_Ka " + rgb_path + "\n")
        f.write("map_Kd " + rgb_path + "\n")


def vete(v, vt):
    return str(v) + "/" + str(vt)


def create_obj(depth_path, back_depth_path, depth_invert, obj_path, mtl_path, mat_name, extrinsic_path, useMaterial, fov):

    c2w = np.load(extrinsic_path)
    w2c = inverse_3x4(c2w)
    
    
    if depth_path.endswith(".npy") or depth_path.endswith(".npz"):
        front_depth = np.load(depth_path).astype(np.float32)
        back_depth = np.load(back_depth_path).astype(np.float32)
    else:
        front_depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
        back_depth = cv2.imread(back_depth_path, -1).astype(np.float32) / 1000.0

    if len(front_depth.shape) > 2 and front_depth.shape[2] > 1:
        print('Expecting a 1D map, but depth map at path %s has shape %r' % (depth_path, front_depth.shape))
        return

    if depth_invert:
        front_depth = 1.0 - front_depth

    w = front_depth.shape[1]
    h = front_depth.shape[0]

    D = (h / 2) / math.tan(fov / 2)

    if max(obj_path.find('\\'), obj_path.find('/')) > -1:
        os.makedirs(os.path.dirname(mtl_path), exist_ok=True)

    with open(obj_path, "w") as f:
        if useMaterial:
            f.write("mtllib " + mtl_path + "\n")
            f.write("usemtl " + mat_name + "\n")

        ids = np.zeros((w * 2, h), int)
        vertex_location = np.ones((w * h * 2 + 2, 3)) * -1
        vid = 1

        for depth_type in ["front", "back"]:
            depth = front_depth if depth_type == "front" else back_depth
            
            for u in range(0, w):
                for v in range(h - 1, -1, -1):

                    d = depth[v, u]

                    if depth_type == "front":
                        ids[u, v] = vid
                        if d == 0.0:
                            ids[u, v] = 0
                    else:
                        ids[w+u, v] = vid
                        if d == 0.0:
                            ids[w+u, v] = 0

                    # TODO: convert this with fovX
                    x = u - w / 2
                    y = v - h / 2
                    z = -D

                    norm = 1 / math.sqrt(x * x + y * y + z * z)

                    t = d / (z * norm)

                    x = -t * x * norm
                    y = t * y * norm
                    z = -t * z * norm
                    x, y, z = w2c @ np.array([x, y, z, 1])
                    
                    if d != 0.0:
                        vertex_location[vid] = np.array([x, y, z])
                    vid += 1
                    f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        # # forward
        for u in range(0, w):
            for v in range(0, h):
                f.write("vt " + str(u / (w * 2)) + " " + str((h - v) / h) + "\n")
                
        # # back
        for u in range(0, w):
            for v in range(0, h):
                f.write("vt " + str(u / (w * 2) + 1/2) + " " + str((h - v) / h) + "\n")

        # for depth_type in ["front", "back"]:
        #     for u in range(0, w):
        #         for v in range(0, h):
        #             if depth_type == "front":
        #                 f.write("vt " + str(u / w / 2) + " " + str(v / h) + "\n")
        #             else:
        #                 f.write("vt " + str((u + w) / (2 * w)) + " " + str(v / h) + "\n")

        T = w2c[:, -1]
        for depth_type in ["front", "back"]:
            depth = front_depth if depth_type == "front" else back_depth
            
            for u in range(0, w - 1):
                for v in range(0, h - 1):
                    if depth_type == "front":
                        v1 = ids[u, v]
                        v2 = ids[u + 1, v]
                        v3 = ids[u, v + 1]
                        v4 = ids[u + 1, v + 1]
                    elif depth_type == "back":
                        v1 = ids[w+u, v]
                        v2 = ids[w+u + 1, v]
                        v3 = ids[w+u, v + 1]
                        v4 = ids[w+u + 1, v + 1]

                    if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                        continue

                    v_loc1 = vertex_location[v1]
                    v_loc2 = vertex_location[v2]
                    v_loc3 = vertex_location[v3]
                    v_loc4 = vertex_location[v4]

                    # if the normal and camera is close to 90 degrees, then skip it
                    normal = np.cross(v_loc1 - v_loc2, v_loc1 - v_loc3)
                    normal /= np.linalg.norm(normal)
                    # check normal is nan
                    ray = (v_loc1 + v_loc2 + v_loc3) / 3.0 - T
                    ray /= np.linalg.norm(ray)
                    angle = math.degrees(math.asin(abs(np.dot(normal, ray))))

                    if angle > 5:
                        f.write("f " + vete(v1, v1) + " " + vete(v2, v2) + " " + vete(v3, v3) + "\n")

                    normal = np.cross(v_loc3 - v_loc2, v_loc3 - v_loc4)
                    normal /= np.linalg.norm(normal)
                    ray = (v_loc3 + v_loc2 + v_loc4) / 3.0 - T
                    ray /= np.linalg.norm(ray)
                    angle = math.degrees(math.asin(abs(np.dot(normal, ray))))
                    if angle > 5:
                        f.write("f " + vete(v3, v3) + " " + vete(v2, v2) + " " + vete(v4, v4) + "\n")


if __name__ == "__main__":
    print("STARTED")
    args = parse_args()
    use_mat = args.rgb_path != ''
    
    if use_mat:
        back_rgb_path = args.rgb_path.replace('.png', '_back.png')
        merged_rgb_path = args.rgb_path.replace('.png', '_merged.png')
        merge_image(args.rgb_path, back_rgb_path, merged_rgb_path)
        create_mtl(args.mtl_path, args.mat_name, merged_rgb_path)
    
    back_depth_path = args.depth_path.replace('depth', 'back_depth')
    create_obj(args.depth_path,
               back_depth_path,
               args.depth_invert,
               args.obj_path,
               args.mtl_path,
               args.mat_name,
               args.extrinsic_path,
               use_mat,
               args.fov)
    print("FINISHED")
