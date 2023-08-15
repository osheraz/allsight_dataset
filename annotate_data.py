import cv2
import numpy as np
import os
import pandas as pd
import json
import math


def get_coords(x, y, angle, imwidth, imheight):
    x1_length = (imwidth - x) / (math.cos(angle) + 1e-6)
    y1_length = (imheight - y) / (math.sin(angle) + 1e-6)
    length = max(abs(x1_length), abs(y1_length))
    endx1 = x + length * math.cos(angle)
    endy1 = y + length * math.sin(angle)

    x2_length = (imwidth - x) / (math.cos(angle + math.pi) + 1e-6)
    y2_length = (imheight - y) / (math.sin(angle + math.pi) + 1e-6)
    length = max(abs(x2_length), abs(y2_length))
    endx2 = x + length * math.cos((angle + math.pi))
    endy2 = y + length * math.sin((angle + math.pi))

    return (int(endx1), int(endy1)), (int(endx2), int(endy2))


class Annotation:

    def __init__(self):

        self.image_list = []

    def annotate(self, i, theta, num, raw_image, indenter, sensor_id, save=False, help_annotate=False):

        w, h = raw_image.shape[:2]
        im2show = cv2.putText(raw_image, i[:-4], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                              cv2.LINE_AA)
        im2show = cv2.putText(im2show, f'Sensor: {sensor_id}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                              1, cv2.LINE_AA)
        im2show = cv2.putText(im2show, f'Indenter: {indenter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 1, cv2.LINE_AA)

        if indenter == 'sphere4':
            rads = {11: 5, 10: 30, 9: 45, 8: 65, 7: 75, 6: 100, 5: 120, 4: 130, 3: 160, 2: 200, 1: 220, 0: 220}
            circle_rads = {11: 20, 10: 20, 9: 25, 8: 30, 7: 30, 6: 35, 5: 40, 4: 45, 3: 45, 2: 50}
        if indenter == 'square' or indenter == 'hexagon' or indenter == 'ellipse':
            rads = {11: 5, 10: 30, 9: 45, 8: 65, 7: 75, 6: 100, 5: 120, 4: 130, 3: 160, 2: 200, 1: 220, 0: 220}
            circle_rads = {11: 20, 10: 20, 9: 25, 8: 30, 7: 30, 6: 35, 5: 40, 4: 45, 3: 45, 2: 50}
        elif indenter == 'sphere5':
            rads = {11: 5, 10: 30, 9: 45, 8: 65, 7: 75, 6: 100, 5: 120, 4: 130, 3: 160, 2: 200, 1: 220, 0: 220}
            circle_rads = {11: 20, 10: 20, 9: 25, 8: 30, 7: 30, 6: 35, 5: 40, 4: 45, 3: 45, 2: 50}
        elif indenter == 'sphere3':
            rads = {11: 5, 10: 30, 9: 45, 8: 65, 7: 75, 6: 100, 5: 120, 4: 130, 3: 160, 2: 200, 1: 220, 0: 220}
            circle_rads = {11: 10, 10: 18, 9: 20, 8: 25, 7: 25, 6: 30, 5: 35, 4: 40, 3: 45, 2: 55, 1: 55}
        else:
            assert 'Problem'

        cr = circle_rads[num]

        if help_annotate:
            lower, upper, length = 0, 360, 30
            degs = [lower + x * (upper - lower) / length for x in range(length)]
            for rad in rads.values():
                im2show = cv2.circle(np.array(im2show), (int(w / 2), int(h / 2)), rad, (160, 160, 160), 1)

            for deg in degs:
                start, end = get_coords(w // 2, h // 2, math.radians(deg), w, h)
                im2show = cv2.line(im2show, (w // 2, h // 2), end, (160, 160, 160), 1)

            for deg in degs:
                for rad in rads.values():
                    pt = (int(w / 2) + math.cos(math.radians(deg)) * rad,
                          int(h / 2) + math.sin(math.radians(deg)) * rad)
                    im2show = cv2.circle(np.array(im2show), (int(pt[0]), int(pt[1])), 3, (255, 255, 255), 1)

        rad = rads[num]
        align_angle = np.pi / 2

        clr = (255, 255, 255)

        center = (int(w / 2) + math.cos(theta + align_angle) * rad,
                  int(h / 2) + math.sin(theta + align_angle) * rad)

        im2show = cv2.circle(np.array(im2show), (int(center[0]), int(center[1])), cr, clr, 1)

        key = -1

        # while key != 27:  # Esc key to stop
        #
        #     coordinateStore = MousePts('contact', im2show, rad)
        #     pts, img = coordinateStore.getpt(1)
        #     if len(pts) == 2:
        #         center = pts[0]
        #         rad = pts[1]
        #     else:
        #         center, rad = (0, 0), 0
        #
        #     if key == 109:
        #         rad += 10
        #     elif key == 110:
        #         rad -= 10
        #     key = cv2.waitKey(0)

        cv2.imshow('contact', im2show.astype(np.uint8))
        key = cv2.waitKey(1)

        save = save
        if save:
            self.image_list.append(im2show)

        return center, cr


if __name__ == "__main__":

    # define the image size, should be consistent with the raw data size
    new_annotate = True
    save = True

    indenter = 'sphere3'
    leds = 'white'
    gel = 'clear'
    date_name = '2023_07_04-07:28:31'

    js_name = 'data_' + date_name
    img_name = 'img_' + date_name
    pc_name = os.getlogin()
    data_path = f'{os.path.dirname(__file__)}/{gel}/{leds}/'

    # JSON_FILE = data_path + f"data/{indenter}/{js_name}/{js_name}_transformed.json"
    # images_path = data_path + f"images/{indenter}/{img_name}/"
    # buffer_paths = [JSON_FILE]

    from glob import glob
    #
    indenter = ['hexagon', 'square', 'ellipse', 'sphere3']
    paths = [f'{os.path.dirname(__file__)}/{gel}/{leds}/data/{ind}' for ind in indenter]
    buffer_paths = []
    for p in paths:
        buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*_transformed.json'))]

    for JSON_FILE in buffer_paths:

        with open('/'.join(JSON_FILE.split('/')[:-1]) + "/summary.json", 'rb') as handle:
            summary = json.load(handle)

        indenter = summary['indenter']
        sensor_id = summary['sensor_id']

        if sensor_id != 13: continue

        df_data = pd.read_json(JSON_FILE).transpose()

        if new_annotate:
            # add new parameters to the table
            #                        (px, py, r)
            df_data['contact_px'] = [[0, 0, 0] for i in range(df_data.shape[0])]
            df_data['annotated'] = False

        df_data = df_data.sort_values(['theta', 'num', 'time'], ascending=[True, True, True])

        ann = Annotation()

        for i in df_data.index:

            name = df_data.loc[i].frame
            img = (cv2.imread(name)).astype(np.uint8)

            if df_data.loc[i].annotated:
                center = (int(df_data.loc[i].contact_px[0]), int(df_data.loc[i].contact_px[1]))
                radius = int(df_data.loc[i].contact_px[2])
                im2show = cv2.circle(np.array(img), center, radius, (0, 40, 0), 2)
                im2show = cv2.putText(im2show, 'annotated!', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                      cv2.LINE_AA)
                img = im2show
                cv2.imshow('contact', im2show.astype(np.uint8))
                key = cv2.waitKey(0)

            try:
                if df_data.loc[i, 'time'] < 0.05:
                    c, r = (0, 0), 1
                else:
                    c, r = ann.annotate(i,
                                        df_data.loc[i, 'theta_transformed'],
                                        df_data.loc[i, 'num'],
                                        img,
                                        indenter,
                                        sensor_id,
                                        save=False)
                    df_data.loc[i, 'annotated'] = True
                df_data.loc[i].contact_px[0], df_data.loc[i].contact_px[1], df_data.loc[i].contact_px[2] = c[0], c[1], r

            except Exception as e:
                print(e, f'{i} is missing annotation')

        print(f'w0w~ Annotation is generated: ({leds} / {gel}) id: {sensor_id} -> ind: {indenter}')

        if save:
            import json

            to_dict = {}
            for index, row in list(df_data.iterrows()):
                to_dict[index] = dict(row)
            with open(r'{}_annotated.json'.format(JSON_FILE[:-5]), 'w') as json_file:
                json.dump(to_dict, json_file, indent=3)

            # import imageio
            # with imageio.get_writer(JSON_FILE[:-12] + ".gif", mode="I") as writer:
            #     for idx, frame in enumerate(ann.image_list):
            #         print("Adding frame to GIF file: ", idx + 1)
            #         writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
