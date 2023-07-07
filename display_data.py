"""

"""

import cv2
import numpy as np
import os
import pandas as pd
import json
from utils import data_for_cylinder_along_z, data_for_sphere_along_z, _structure, _diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from utils import Arrow3D
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from transformations import quaternion_matrix


class Display:

    def __init__(self):
        self.image_list = []

        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.autoscale(enable=True, axis='both', tight=True)

        self.ax.set_xlim((-0.014, 0.014))
        self.ax.set_ylim((-0.014, 0.014))
        self.ax.set_zlim((0.0, 0.03))

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        Xc, Yc, Zc = data_for_cylinder_along_z(center_x=0, center_y=0, radius=0.012, height_z=0.016)
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
        Xc, Yc, Zc = data_for_sphere_along_z(center_x=0, center_y=0, radius=0.012, height_z=0.016)
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

        self.hl, = self.ax.plot3D([0], [0], [0], '-')

        self.tm = TransformManager()

    def visualize(self, i, cp, pose, ft, raw_image, ref_img, save=False):
        # cv2.waitKey(0)
        raw_copy = raw_image.copy()
        im2show = cv2.putText(raw_image, i, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        c = (255, 255, 255)

        im2show = cv2.circle(np.array(im2show), (int(cp[0]), int(cp[1])), cp[2], c, 1)
        img_stct = _structure(raw_copy).astype(np.uint8)
        cv2.imshow('contact', np.concatenate((im2show.astype(np.uint8),
                                              np.dstack((img_stct, img_stct, img_stct)),
                                              _diff(raw_copy, ref_img).astype(np.uint8)), axis=1))

        # object2cam = pt.transform_from_pq(np.hstack((pose[0][0:3],
        #                                              pr.quaternion_wxyz_from_xyzw(pose[1][0:4]))))
        # self.tm.add_transform("object" + str(i), "camera", object2cam)
        # self.ax = self.tm.plot_frames_in("camera", s=0.002, show_name=False)

        # self.ax.scatter(*pose[0][0:3])

        scale = 1000
        pose = pose[0]
        # print(ft[:3])
        # update_arrow(self.hl, ([pose[0], pose[0] + ft[0] / scale],
        #                        [pose[1], pose[1] + ft[1] / scale],
        #                        [pose[2], pose[2] + ft[2] / scale])
        #              )
        # plt.draw()
        # plt.pause(0.00001)

        if save:
            self.image_list.append(im2show)

        cv2.waitKey(1)

def convert_quat_xyzw_to_wxyz(q):
    q[0], q[1], q[2], q[3] = q[3], q[0], q[1], q[2]
    return q

if __name__ == "__main__":

    # define the image size, should be consistent with the raw data size
    # w, h = 640, 480
    save = False

    indenter = 'sphere4'
    leds = 'rrrgggbbb'  # 'rgbrgbrgb
    gel = 'markers'
    date_name = '2023_05_18-12:22:16'  # '2023_01_29-04:20:49' # '2023_01_29-01:51:55'
    js_name = 'data_' + date_name
    img_name = 'img_' + date_name
    pc_name = os.getlogin()

    data_path = f'/home/{pc_name}/catkin_ws/src/allsight/dataset/{gel}/{leds}/'
    JSON_FILE = data_path + f"data/{indenter}/{js_name}/{js_name}_transformed_annotated.json"
    images_path = data_path + f"images/{indenter}/{img_name}/"

    df_data = pd.read_json(JSON_FILE).transpose()

    df_data = df_data.sort_values(['theta', 'num', 'time'], ascending=[True, True, True])
    # df_data = df_data[df_data.time > 1.0]  # train only over touching samples!

    disp = Display()

    for i in df_data.index:
        name = df_data.loc[i].frame.replace('osher', pc_name)
        ref_name = df_data.loc[i].ref_frame.replace('osher', pc_name)

        # img = (cv2.imread(name) * mask3).astype(np.uint8)
        img = (cv2.imread(name)).astype(np.uint8)
        ref_img = (cv2.imread(ref_name)).astype(np.uint8)

        rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(df_data['pose_transformed'][i][1][0:4]))
        fxyz = df_data['ft_ee_transformed'][i][0:3]
        fxyz = np.dot(rot[:3, :3], fxyz)

        disp.visualize(i,
                       df_data.loc[i, 'contact_px'],
                       df_data.loc[i, 'pose_transformed'],
                       fxyz,
                       img,
                       ref_img)

    print('w0w~ Display is generated')

    if save:
        import json

        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_annotated.json'.format(JSON_FILE[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)

        # with imageio.get_writer(JSON_FILE[:-12] + ".gif", mode="I") as writer:
        #     for idx, frame in enumerate(disp.image_list):
        #         print("Adding frame to GIF file: ", idx + 1)
        #         writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
