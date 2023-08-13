import numpy as np
import pandas as pd
from transformations import translation_matrix, rotation_matrix, translation_from_matrix, quaternion_matrix, quaternion_from_matrix
import os
from glob import glob
import json

# TODO SUPER NOTE !! in the transformations package, Quaternions w+ix+jy+kz are represented as [w, x, y, z].

def convert_quat_xyzw_to_wxyz(q):
    q[0], q[1], q[2], q[3] = q[3], q[0], q[1], q[2]
    return q


def convert_quat_wxyz_to_xyzw(q):
    q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
    return q

origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

# change depend on the data you use
pc_name = os.getlogin()
gel = 'clear'
leds = 'white'
# Specific buffer
# indenter = 'sphere4'
# data_name = 'data_2023_07_04-07:28:31'
# JSON_FILE = f"{os.path.dirname(__file__)}/{gel}/{leds}/data/{indenter}/{data_name}/{data_name}.json"
# buffer_paths = [JSON_FILE]

# Multiple buffer
indenter = ['sphere3', 'square', 'ellipse', 'hexagon']
paths = [f"{os.path.dirname(__file__)}/{gel}/{leds}/data/{ind}" for ind in indenter]
buffer_paths = []
for p in paths:
    buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]

buffer_paths = [p for p in buffer_paths if ('transformed' not in p) and
                                           ('final' not in p) and
                                           ('summary' not in p)
                ]


j_i = 0

for JSON_FILE in buffer_paths:

    j_i += 1
    print(f'transforming dataset: {JSON_FILE[-58:]} \t {j_i}/{len(buffer_paths)}')

    df_data = pd.read_json(JSON_FILE).transpose()

    check = 'pose_ee' in df_data.keys()

    print(f'df length: {df_data.shape}, is ee pose recorded?: {check}')

    df_data['theta_transformed'] = 0
    df_data['depth'] = 0

    df_data = df_data.assign(pose_transformed=[[[0, 0, 0], [0, 0, 0, 0]] for _ in df_data.index])
    df_data = df_data.assign(ft_transformed=[[0, 0, 0, 0, 0, 0] for _ in df_data.index])

    if 'pose_ee' in df_data.keys():
        df_data = df_data.assign(pose_ee_transformed=[[[0, 0, 0], [0, 0, 0, 0]] for _ in df_data.index])
        df_data = df_data.assign(ft_ee_transformed=[[0, 0, 0, 0, 0, 0] for _ in df_data.index])

    '''
    # Finger properties
    z_down, z_up = (0.7071068, 0, -0.7071068, 0), (1, 0, 0, 0)
    trans_finger_origin, rot_finger_origin, rot_finger_up = (0, 0, 0), z_down, z_up
    T_world_fingerOrigin rotation of -90 degrees in y axis - pointing towards the robot.
    T_world_fingerUp - that's the origin
    T_world_fingerOrigin = np.dot(translation_matrix(trans_finger_origin), quaternion_matrix(rot_finger_origin))
    T_world_fingerUp = np.dot(translation_matrix(trans_finger_origin), quaternion_matrix(rot_finger_up))
    '''

    for i in df_data.index:

        # get pressing pose w.r.t finger origin
        # pose is measured in T_world_fingerOrigin (-90 degrees in y axis)
        trans, rot = df_data.loc[i].pose[0], df_data.loc[i].pose[1]
        trans_mat, rot_mat = translation_matrix(trans), quaternion_matrix(convert_quat_xyzw_to_wxyz(rot))
        T_fingerOrigin_press_z_right = np.dot(trans_mat, rot_mat)

        # rotate everything for z_up ( +90 over y axis)
        Ry = rotation_matrix(- np.pi / 2, yaxis)
        T_fingerOrigin_press_zup = np.matmul(T_fingerOrigin_press_z_right, Ry)

        # convert theta from 0 - 1 to 0 - 2* pi, with interpolating from min/max motor setting
        q_to_angle = np.interp(df_data.loc[i].theta, [0, 1.0], [0.002, 0.99])
        df_data.loc[i, "theta_transformed"] = np.interp(q_to_angle, [0, 1], [0, 2 * np.pi])  # theta_transformed

        # rotate around the finger axis by theta
        Rz = rotation_matrix(df_data.loc[i].theta_transformed, zaxis)
        T_fingerOrigin_press_rotate_q_zup = np.matmul(Rz, T_fingerOrigin_press_zup)

        # append to the data table - press point ( z pointing towards the finger) w.r.t finger origin (z up).
        trans_q = translation_from_matrix(T_fingerOrigin_press_rotate_q_zup)
        rot_q = quaternion_from_matrix(T_fingerOrigin_press_rotate_q_zup)

        df_data['pose_transformed'][i][0] = trans_q.tolist()
        df_data['pose_transformed'][i][1] = convert_quat_wxyz_to_xyzw(rot_q.tolist())

        T_fingerOrigin_press = T_fingerOrigin_press_rotate_q_zup
        # ^ is the transformation matrix from press point to the finger origin ( z pointing towards the finger)

        # rotate the force to the normal orientation
        # force is already measured w.r.t ee pose and orientation.
        Rn = T_fingerOrigin_press[:3, :3]

        f_ = np.dot(Rn, df_data.loc[i, "ft"][:3])
        tau_ = np.dot(Rn, df_data.loc[i, "ft"][3:])
        df_data['ft_transformed'][i][:6] = np.hstack((f_, tau_)).tolist()
        # ^ ft_transformed is the force projected to the xyz plane for easy plotting.

        ###########################################################
        # def projection(p, a):
        #     lambda_val = np.dot(p, a) / np.dot(a, a)
        #     return p - lambda_val * a
        #
        # n = Rn[-1]
        # f = np.array(df_data.loc[i, "ft"][:3])
        # f_to_plane = np.linalg.norm(f) * projection(f / np.linalg.norm(f), n)
        ##############################################

        if 'pose_ee' in df_data.keys():
            # get ee pressing pose w.r.t finger origin
            # ee pose is measured in T_world_fingerOrigin (-90 degrees in y axis)

            trans_ee, rot_ee = df_data.loc[i].pose_ee[0], df_data.loc[i].pose_ee[1]
            trans_mat_ee, rot_mat_ee = translation_matrix(trans_ee), quaternion_matrix(
                convert_quat_xyzw_to_wxyz(rot_ee))
            T_fingerOrigin_press_z_right_ee = np.dot(trans_mat_ee, rot_mat_ee)

            # rotate everything for z_up ( +90 over y axis)
            T_fingerOrigin_press_zup_ee = np.matmul(T_fingerOrigin_press_z_right_ee, Ry)

            # rotate around the finger axis by theta
            T_fingerOrigin_press_rotate_q_ee_zup = np.matmul(Rz, T_fingerOrigin_press_zup_ee)

            # append to the data table - ee press point ( z pointing towards the finger) w.r.t finger origin (z up).
            trans_q_ee = translation_from_matrix(T_fingerOrigin_press_rotate_q_ee_zup)
            rot_q_ee = quaternion_from_matrix(T_fingerOrigin_press_rotate_q_ee_zup)

            df_data['pose_ee_transformed'][i][0] = trans_q_ee.tolist()
            df_data['pose_ee_transformed'][i][1] = convert_quat_wxyz_to_xyzw(rot_q_ee.tolist())

            df_data.loc[i, "depth"] = np.linalg.norm(trans_q_ee - trans_q)

            T_fingerOrigin_press_ee = T_fingerOrigin_press_rotate_q_ee_zup

            # calc the force w.r.t the finger surface
            r_0 = T_fingerOrigin_press_ee[:3, :3]
            # ^ is the rotation matrix from ee pose to the finger origin ( z pointing towards the finger)
            r_1 = T_fingerOrigin_press[:3, :3]
            # ^ is the rotation matrix from press point to the finger origin ( z pointing towards the finger)

            # rotation matrix that rotates r_0 to r_1
            r0_to_r1 = r_1.dot(r_0.T)

            f_ee = np.dot(r0_to_r1, df_data.loc[i, "ft"][:3])
            tau_ee = np.dot(r0_to_r1, df_data.loc[i, "ft"][3:])
            df_data['ft_ee_transformed'][i][:6] = np.hstack((f_ee, tau_ee)).tolist()  # (normal to surface)

            #######################################
            # T_rot_fingerOrigin_press = np.matmul(T_fingerOrigin_press_rotate_q, Ry)
            # rot_f = T_rot_fingerOrigin_press[:3, :3]
            # f_ = np.dot(rot_f, df_data.loc[i, "ft"][:3])
            # tau_ = np.dot(rot_f, df_data.loc[i, "ft"][3:])
            # df_data['ft_ee_transformed'][i][:6] = np.hstack((f_, tau_)).tolist()
            #######################################

    save = True
    if save:
        import json

        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)

    plot = True

    if plot:

        # do some plotting
        import matplotlib.pyplot as plt
        from pytransform3d import rotations as pr
        from pytransform3d import transformations as pt
        from pytransform3d.transform_manager import TransformManager
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        df_data = pd.read_json('{}_transformed.json'.format(JSON_FILE[:-5])).transpose()

        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(df_data.pose_transformed)):
            pxyz = df_data['pose_transformed'][i][0]
            ax.scatter(pxyz[0], pxyz[1], pxyz[2], c='black')
            f = df_data['pose'][i][0]
            ax.scatter(f[0], f[1], f[2], c='red')

        ax.view_init(90, 90)
        fig.savefig(JSON_FILE[:-5] + '_test', dpi=200, bbox_inches='tight')

        from numpy import *
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d


        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)


        tm = TransformManager()

        amnt = 10 if len(df_data) < 6000 else 25
        for i in range(0, len(df_data['pose_transformed']), amnt):
            object2cam = pt.transform_from_pq(np.hstack((df_data['pose_transformed'][i][0][0:3],
                                                         pr.quaternion_wxyz_from_xyzw(
                                                             df_data['pose_transformed'][i][1][0:4]))))

            tm.add_transform("object" + str(i), "camera", object2cam)

        ax = tm.plot_frames_in("camera", s=0.0015, show_name=False)

        scale = 2000
        for i in range(0, len(df_data['pose_transformed']), 5):
            pxyz = df_data['pose_transformed'][i][0][0:3]
            fxyz = df_data['ft_transformed'][i][0:3]

            a = Arrow3D([pxyz[0], pxyz[0] + fxyz[0] / scale], [pxyz[1], pxyz[1] + fxyz[1] / scale],
                        [pxyz[2], pxyz[2] + fxyz[2] / scale], mutation_scale=5,
                        lw=0.3, arrowstyle="-|>", color="k")
            ax.add_artist(a)

        ax.set_xlim((-0.014, 0.014))
        ax.set_ylim((-0.014, 0.014))
        ax.set_zlim((0.0, 0.03))
        fig.savefig(JSON_FILE[:-5] + '_transform_pose', dpi=200, bbox_inches='tight')
        ax.view_init(90, 90)
        fig.savefig(JSON_FILE[:-5] + '_transform_pose_top', dpi=200, bbox_inches='tight')
        ax.view_init(0, 0)
        fig.savefig(JSON_FILE[:-5] + '_transform_pose_side', dpi=200, bbox_inches='tight')

        if 'pose_ee' in df_data.keys():

            tm = TransformManager()
            for i in range(0, len(df_data['pose_ee_transformed']), amnt):
                object2cam = pt.transform_from_pq(np.hstack((df_data['pose_ee_transformed'][i][0][0:3],
                                                             pr.quaternion_wxyz_from_xyzw(
                                                                 df_data['pose_ee_transformed'][i][1][0:4]))))

                tm.add_transform("object" + str(i), "camera", object2cam)

            ax = tm.plot_frames_in("camera", s=0.0015, show_name=False)

            scale = 2000

            for i in range(0, len(df_data['pose_ee_transformed']), 5):
                pxyz = df_data['pose_ee_transformed'][i][0][0:3]
                rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(df_data['pose_transformed'][i][1][0:4]))
                fxyz = df_data['ft_ee_transformed'][i][0:3]
                fxyz = np.dot(rot[:3, :3], fxyz)

                f_norm = np.linalg.norm(fxyz)

                a = Arrow3D([pxyz[0], pxyz[0] + fxyz[0] / scale], [pxyz[1], pxyz[1] + fxyz[1] / scale],
                            [pxyz[2], pxyz[2] + fxyz[2] / scale], mutation_scale=5,
                            lw=0.3, arrowstyle="-|>", color="k")
                ax.add_artist(a)

            ax.set_xlim((-0.014, 0.014))
            ax.set_ylim((-0.014, 0.014))
            ax.set_zlim((0.0, 0.03))
            # ax.view_init(90, 90)
            fig.savefig(JSON_FILE[:-5] + '_transform_pose_ee', dpi=200, bbox_inches='tight')
            ax.view_init(90, 90)
            fig.savefig(JSON_FILE[:-5] + '_transform_pose_ee_top', dpi=200, bbox_inches='tight')
            ax.view_init(0, 0)
            fig.savefig(JSON_FILE[:-5] + '_transform_pose_ee_side', dpi=200, bbox_inches='tight')

        ############## FORCE

        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 1, 2)

        force = np.array([df_data.iloc[idx].ft_ee_transformed for idx in range(df_data.shape[0])])[:, :3]
        force_mag_true = np.linalg.norm(force, axis=1)

        ax1.hist(force[:, 0], density=True, bins=30)  # density=False would make counts
        ax2.hist(force[:, 1], density=True, bins=30)  # density=False would make counts
        ax3.hist(force[:, 2], density=True, bins=30)  # density=False would make counts
        ax4.hist(force_mag_true, density=True, bins=30)  # density=False would make counts

        if dir is not None:
            plt.tight_layout()
            fig.savefig(JSON_FILE[:-5] + '_plot_force.png', dpi=200, bbox_inches='tight')

        ############## TORQUE

        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 1, 2)

        torque = np.array([df_data.iloc[idx].ft_ee_transformed for idx in range(df_data.shape[0])])[:, 3:]
        torque_mag_true = np.linalg.norm(torque, axis=1)

        ax1.hist(torque[:, 0], density=True, bins=30)  # density=False would make counts
        ax2.hist(torque[:, 1], density=True, bins=30)  # density=False would make counts
        ax3.hist(torque[:, 2], density=True, bins=30)  # density=False would make counts
        ax4.hist(torque_mag_true, density=True, bins=30)  # density=False would make counts

        if dir is not None:
            plt.tight_layout()
            fig.savefig(JSON_FILE[:-5] + '_plot_torque.png', dpi=200, bbox_inches='tight')

        plt.close('all')
    print('Finished to transform the data.\n')