import numpy as np
import cv2
import os
import json
import random
from shapely.geometry import Polygon
from datetime import datetime
import argparse

DEBUG_P = False
if not DEBUG_P:
    from segment_anything import sam_model_registry, SamPredictor
    checkpoint = "./sam_ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
VIS = True
VIS_POINT_R = 3
P_THICKNESS = 1
KEEP_ONE = True

def main(comm, point_n, mask_n, current_time_str):
    NUM_POS_P = point_n
    NUM_NEG_P = point_n
    ITER = 0
    POINT_ADD_N = 1
    MAX_ATT = 50000
    TINY = 1e-10
    COMM = comm
    FEA = "DTS"
    img_root = "./IMG/" + COMM
    poly_root = "./ANNO/" + COMM
    if mn >= 0:
        SAVE_STR = "_P" + str(NUM_POS_P) + "_N" + str(NUM_NEG_P) + "_MN" + str(mask_n)
    else:
        SAVE_STR = "_P" + str(NUM_POS_P) + "_N" + str(NUM_NEG_P)
    TEMP_STR = "./Results/" + current_time_str + "_Random/"
    if not os.path.exists("./Results/"):
        os.mkdir("./Results/")
    if not os.path.exists(TEMP_STR):
        os.mkdir(TEMP_STR)
    txt_path = os.path.join(TEMP_STR, COMM + "_" + FEA) + "/sam_zs_log_vit_h" + SAVE_STR + ".txt"
    def calculate_iou(poly1, poly2):
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        iou = intersection_area / union_area
        return iou

    def calculate_miou(polygons1, polygon2):
        num_polygons = len(polygons1)
        ious = {}
        for i in range(num_polygons):
            if polygons1[i].shape[0] < 3:
                continue
            poly1 = Polygon(polygons1[i])
            poly2 = Polygon(polygon2)
            try:
                iou = calculate_iou(poly1, poly2)
            except:
                iou = calculate_iou(poly1.buffer(0), poly2.buffer(0))
            ious[i] = iou
        return ious

    def eval_poly(mask, poly_gt):
        color = np.array([255, 255, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        pred_mask = cv2.cvtColor(mask_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        poly_preds = []
        for cnt in contours:
            poly_preds.append(np.squeeze(cnt).astype(int))
        return calculate_miou(poly_preds, poly_gt), poly_preds, poly_gt

    def write_mask(ori_image, pred_mask_single, gt_poly, iter_pred_mask, iter_gt_mask, color_i):
        with open('color_list_200.json', 'r') as j:
            color_list = json.loads(j.read())
        h, w = pred_mask_single.shape[-2:]
        mask_image_pred = pred_mask_single.reshape(h, w, 1) * np.array([
            color_list[color_i][0], color_list[color_i][1], color_list[color_i][2],
        ]).reshape(1, 1, -1)
        # print(mask_image_pred.shape, iter_pred_mask.shape)
        iter_pred_mask = cv2.add(mask_image_pred.astype(int), iter_pred_mask.astype(int))
        # print(gt_poly.shape)
        cv2.fillPoly(iter_gt_mask, np.int32([gt_poly]),
                     (color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]))
        # res_image = cv2.addWeighted(ori_image, 0.7, mask_image, 0.3, gamma=0, dtype=cv2.CV_64F)
        return iter_pred_mask, iter_gt_mask

    def get_triangle(center_point, shape_size):
        triangle_points = np.array([
            center_point,
            (center_point[0] - shape_size, center_point[1] + shape_size),
            (center_point[0] + shape_size, center_point[1] + shape_size)
        ], np.int32)
        return triangle_points

    def get_star(center_point, shape_size):
        star_points = []
        for i in range(5):
            angle = 2 * np.pi * i / 5
            x = int(center_point[0] + shape_size * np.cos(angle))
            y = int(center_point[1] - shape_size * np.sin(angle))
            star_points.append((x, y))
        star_points = np.array(star_points, np.int32)
        return star_points

    def write_bdy_pts(ori_image, pred_mask_single, gt_poly, iter_pred_mask, color_i, iou_dict,
                      old_pos_pts, old_neg_pts, new_pos_pts):
        with open('color_list_200.json', 'r') as j:
            color_list = json.loads(j.read())
        h, w = pred_mask_single.shape[-2:]
        mask_image_pred = pred_mask_single.reshape(h, w, 1) * np.array([(255, 255, 255)]).reshape(1, 1, -1)
        pred_mask = cv2.cvtColor(mask_image_pred.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        poly_preds = []
        if KEEP_ONE:
            res_key = max(iou_dict, key=lambda x: iou_dict[x])
            poly_preds.append(np.squeeze(contours[res_key]).astype(int))
        else:
            for cnt in contours:
                poly_preds.append(np.squeeze(cnt).astype(int))
        for pp in poly_preds:
            try:
                cv2.polylines(iter_pred_mask, np.int32([pp]),
                              isClosed=True,
                              color=(color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]),
                              thickness=5)
                if VIS_POINT_R > 0:
                    for p in old_pos_pts:
                        cv2.polylines(iter_pred_mask, np.int32([get_triangle(tuple(p), VIS_POINT_R)]), isClosed=True,
                                      color=(color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]),
                                      thickness=P_THICKNESS)
                    for p in old_neg_pts:
                        cv2.circle(iter_pred_mask, tuple(p), VIS_POINT_R,
                                   color=(color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]),
                                   thickness=P_THICKNESS)
                    if ITER > 1:
                        for p in new_pos_pts:
                            cv2.polylines(iter_pred_mask, np.int32([get_star(tuple(p), VIS_POINT_R)]), isClosed=True,
                                          color=(
                                          color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]),
                                          thickness=P_THICKNESS)

            except:
                print(pp, 'error!')
        write_one_str = ("Color_idx: " + str(color_i) + "| pos_pts: " + str(len(old_pos_pts)) +
                         "| neg_pts: " + str(len(old_neg_pts)) +
                         "| iou: " + str(round(max(iou_dict.values()), 5)))
        print(write_one_str)
        f = open(txt_path, 'a')
        f.write(write_one_str + '\n')
        f.close()

        cv2.polylines(iter_pred_mask, np.int32([gt_poly]),
                      isClosed=True,
                      color=(color_list[color_i][0], color_list[color_i][1], color_list[color_i][2]),
                      thickness=1, lineType=cv2.LINE_AA)
        return iter_pred_mask

    def find_center_in_polygon(polygon, h, w):
        polygon_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)
        contours, _ = cv2.findContours(polygon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(polygon_contour)
        center_x = int(M["m10"] / (M["m00"] + TINY))
        center_y = int(M["m01"] / (M["m00"] + TINY))
        while cv2.pointPolygonTest(polygon, (center_x, center_y), False) <= 0:
            n_p = polygon.shape[0]
            idx3 = random.sample(range(n_p), 3)
            x1, y1 = polygon[idx3[0], :]
            x2, y2 = polygon[idx3[1], :]
            x3, y3 = polygon[idx3[2], :]
            center_x = int((x1 + x2 + x3) / 3)
            center_y = int((y1 + y2 + y3) / 3)
        assert cv2.pointPolygonTest(polygon, (center_x, center_y), False) > 0

        return (center_x, center_y)

    def generate_points_in_polygon_3(polygon, num_points, image):
        x, y, w, h = cv2.boundingRect(polygon)
        hh, ww = image.shape[0], image.shape[1]
        (c_x, c_y) = find_center_in_polygon(polygon, hh, ww)
        points = []
        attempts = 0
        while len(points) < num_points and attempts < MAX_ATT:
            random_point = (int(np.random.normal(loc=c_x, scale=1e+2)),
                            int(np.random.normal(loc=c_y, scale=1e+2)))
            attempts += 1
            if cv2.pointPolygonTest(polygon, random_point, False) > 0:
                points.append(random_point)

        return points

    def generate_points_outside_polygon_3(polygon, num_points):
        x, y, w, h = cv2.boundingRect(polygon)
        points = []

        while len(points) < num_points:
            random_point = (int(np.random.uniform(x, x + w)),
                            int(np.random.uniform(y, y + h)))
            if cv2.pointPolygonTest(polygon, random_point, False) < 0:
                points.append(random_point)

        return points

    def filter_preds_including_pos_pts(polygon, points):
        points_f = []
        for p in points:
            if cv2.pointPolygonTest(polygon, p, False) > 0:
                points_f.append(p)
        return points_f

    def inference_one_img(img_path, point_path, ip, pp):
        # loading image
        image = cv2.imread(img_path)
        if not DEBUG_P:
            predictor.set_image(image)

        one_mIoU = []
        # gather all the polys
        with open(point_path, 'r') as j:
            poly_list = json.loads(j.read())
        if VIS:
            color_pred_mask = np.copy(image)
            color_gt_mask = np.copy(image)
            color_idx = 0
        for pl in poly_list:
            # point prompts
            pos_points = generate_points_in_polygon_3(np.array(pl, np.int32), NUM_POS_P, image)
            pos_labels = []
            real_pos_num = np.array(pos_points).shape[0]
            for i in range(real_pos_num):
                pos_labels.append(1)
            neg_points = generate_points_outside_polygon_3(np.array(pl), NUM_NEG_P)
            neg_labels = []
            real_neg_num = np.array(neg_points).shape[0]
            for i in range(real_neg_num):
                neg_labels.append(0)

            if VIS_POINT_R > 0:
                old_pos_pts = []
                old_pos_pts.extend(pos_points)
                old_neg_pts = neg_points
                new_pos_pts = []

            if DEBUG_P:
                one_mIoU.append(-1.0)
            else:
                masks, scores, logits = predictor.predict(
                    point_coords=np.concatenate((np.array(pos_points), np.array(neg_points)), axis=0),
                    point_labels=np.concatenate((np.array(pos_labels), np.array(neg_labels)), axis=0),
                    multimask_output=mn >= 0,
                )
                # show results
                iou_dict, poly_preds, poly_gt = eval_poly(masks[mask_n], np.array(pl))

                # vis. all
                if len(iou_dict.keys()) > 0:
                    one_mIoU.append(np.max(list(iou_dict.values())) if KEEP_ONE else np.mean(list(iou_dict.values())))
                    if VIS:
                        if VIS_POINT_R > 0:
                            color_pred_mask = write_bdy_pts(
                                image, masks[mask_n], np.array(pl), color_pred_mask, color_idx, iou_dict,
                                old_pos_pts, old_neg_pts, new_pos_pts
                            )
                        else:
                            color_pred_mask = write_bdy(
                                image, masks[mask_n], np.array(pl), color_pred_mask, color_idx, iou_dict
                            )
                        color_idx += 1
                else:
                    one_mIoU.append(0.0)
        if VIS:
            if not os.path.exists(os.path.join(TEMP_STR, COMM + "_" + FEA) +
                                  "/pred_color_h" + SAVE_STR):
                os.mkdir(os.path.join(TEMP_STR, COMM + "_" + FEA) + "/pred_color_h" + SAVE_STR)
            cv2.imwrite(os.path.join(os.path.join(TEMP_STR, COMM + "_" + FEA) +
                                     "/pred_color_h" + SAVE_STR, ip), color_pred_mask)
        return np.mean(one_mIoU)

    img_paths = os.listdir(img_root)
    img_paths.sort()
    poly_paths = os.listdir(poly_root)
    poly_paths.sort()
    poly_id_list = []
    for pp in poly_paths:
        poly_id_list.append(pp.split("__")[0])
    final_mIoU = []
    if not os.path.exists(os.path.join(TEMP_STR, COMM + "_" + FEA)):
        os.mkdir(os.path.join(TEMP_STR, COMM + "_" + FEA))
    p_i = 0
    for i in range(len(img_paths)):
        if img_paths[i].split('.')[0] in poly_id_list:
            # print(img_paths[i].split('.')[0])
            ip = img_paths[i]
            pp = poly_paths[p_i]
            pp = img_paths[i].split('.')[0] + '__' + FEA + '.json'
            print(ip, pp)
            one_mIoU = inference_one_img(os.path.join(img_root, ip), os.path.join(poly_root, pp), ip, pp)
            print(ip, pp, one_mIoU)
            f = open(txt_path, 'a')
            f.write(ip + " " + pp + " " + str(one_mIoU) + '\n' + '\n')
            f.close()
            final_mIoU.append(one_mIoU)
            p_i += 1
    print("--------------------------------------------------------")
    print("mIoU:", np.mean(final_mIoU))
    print("Mining Num:", len(poly_paths))
    f = open(txt_path, 'a')
    f.write("mIoU: " + str(np.mean(final_mIoU)) + '\n')
    f.write("Mining Num: " + str(len(poly_paths)) + '\n')
    f.close()
    print("One job done!")

mn_list = [-1]
current_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
parser = argparse.ArgumentParser(description="")
parser.add_argument('--comm_list', type=str, nargs='+',
                    help="The commodity you want to test with MO-SAM, choose from: Co, Li, REE, PGE.")
parser.add_argument('--pn_list', type=int, nargs='+', help="The number of positive and negative point prompt.")
args = parser.parse_args()

for c in args.comm_list:
    for pn in args.pn_list:
        for mn in mn_list:
            print('start', c, pn, mn)
            main(comm=c, point_n=pn, mask_n=mn, current_time_str=current_time_str)
