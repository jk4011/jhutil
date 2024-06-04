# https://github.com/CapObvios/Depth-Map-Visualizer/blob/master/DepthToObj.py
import argparse
import numpy as np
import cv2
import math
import os
from jhutil import inverse_3x4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--depthPath', dest='depthPath',
                        help='depth map path',
                        default='objaverse-rendering/example/example_obj/000_depth.npy', type=str)
    parser.add_argument('--depthInvert', dest='depthInvert',
                        help='Invert depth map',
                        default=False, action='store_true')
    parser.add_argument('--texturePath', dest='texturePath',
                        help='corresponding image path',
                        default='objaverse-rendering/example/example_obj/000.png', type=str)
    parser.add_argument('--objPath', dest='objPath',
                        help='output path of .obj file',
                        default='model.obj', type=str)
    parser.add_argument('--mtlPath', dest='mtlPath',
                        help='output path of .mtl file',
                        default='model.mtl', type=str)
    parser.add_argument('--matName', dest='matName',
                        help='name of material to create',
                        default='colored', type=str)
    parser.add_argument('--extrinsicPath', dest='extrinsicPath',
                        help='path to extrinsic matrix',
                        default='objaverse-rendering/example/example_obj/000.npy', type=str)
    parser.add_argument('--fov', dest='fov',
                        help='field of view',
                        default=0.8575560548920328, type=float)
    args = parser.parse_args()
    return args


def create_mtl(mtlPath, matName, texturePath):
    if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(mtlPath, "w") as f:
        f.write("newmtl " + matName + "\n")
        f.write("Ns 10.0000\n")
        f.write("d 1.0000\n")
        f.write("Tr 0.0000\n")
        f.write("illum 2\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("map_Ka " + texturePath + "\n")
        f.write("map_Kd " + texturePath + "\n")


def vete(v, vt):
    return str(v) + "/" + str(vt)


def create_obj(depthPath, depthInvert, objPath, mtlPath, matName, extrinsicPath, useMaterial, fov):

    c2w = np.load(extrinsicPath)
    w2c = inverse_3x4(c2w)
    
    
    if depthPath.endswith(".npy") or depthPath.endswith(".npz"):
        depth = np.load(depthPath).astype(np.float32)
    else:
        depth = cv2.imread(depthPath, -1).astype(np.float32) / 1000.0

    if len(depth.shape) > 2 and depth.shape[2] > 1:
        print('Expecting a 1D map, but depth map at path %s has shape %r' % (depthPath, depth.shape))
        return

    if depthInvert:
        depth = 1.0 - depth

    w = depth.shape[1]
    h = depth.shape[0]

    D = (depth.shape[0] / 2) / math.tan(fov / 2)

    if max(objPath.find('\\'), objPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)

    with open(objPath, "w") as f:
        if useMaterial:
            f.write("mtllib " + mtlPath + "\n")
            f.write("usemtl " + matName + "\n")

        ids = np.zeros((depth.shape[1], depth.shape[0]), int)
        vertex_location = np.ones((depth.shape[1] * depth.shape[0] + 2, 3)) * -1
        vid = 1

        for u in range(0, w):
            for v in range(h - 1, -1, -1):

                d = depth[v, u]

                ids[u, v] = vid
                if d == 0.0:
                    ids[u, v] = 0

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
                # write with color
                # f.write("v " + str(x) + " " + str(y) + " " + str(z) + " " + str(img[v,u]) + " " + str(img[v,u]) + " " + str(img[v,u]) + "\n")

        for u in range(0, depth.shape[1]):
            for v in range(0, depth.shape[0]):
                f.write("vt " + str(u / depth.shape[1]) + " " + str(v / depth.shape[0]) + "\n")

        T = w2c[:, -1]
        for u in range(0, depth.shape[1] - 1):
            for v in range(0, depth.shape[0] - 1):

                v1 = ids[u, v]; v2 = ids[u + 1, v]; v3 = ids[u, v + 1]; v4 = ids[u + 1, v + 1]

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
    useMat = args.texturePath != ''
    if useMat:
        create_mtl(args.mtlPath, args.matName, args.texturePath)
    create_obj(args.depthPath,
               args.depthInvert,
               args.objPath,
               args.mtlPath,
               args.matName,
               args.extrinsicPath,
               useMat,
               args.fov)
    print("FINISHED")
