# -*- coding: utf-8 -*-
import subprocess
import sys
import time
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
import craft_utils
import imgproc
from craft import CRAFT
from collections import OrderedDict
import copy
import platform
import os

sys.path.insert(0, os.path.abspath('./module/'))

def preprocessing(img_path):
    # example: img_path='/content/drive/My Drive/ocr_demo_code/test_imgs/test_1708/0.jpg'
    pre_img_path = './imageStorage/'+'pre_' + \
        os.path.basename(img_path)  # pre_0.jpg
    if platform.system() == 'Darwin':
        # ../../imgtxtenh/imgtxtenh 0.jpg -p pre_0.jpg
        command = './imgtxtenh/imgtxtenh ' + ' -p -t sauvolaSdMax -k 0.36 -s 0.5' + " " + \
            img_path + " " + pre_img_path
        print(command)
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        # ../../imgtxtenh/imgtxtenh 0.jpg -p pre_0.jpg
        command = './imgtxtenh/imgtxtenh ' + " " + \
            img_path + ' -p -t sauvolaSdMax -k 0.36 -s 0.5' + " " +\
            pre_img_path

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    p.wait()
    print(pre_img_path, img_path)
    return pre_img_path


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_CUBIC, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    t1 = time.time() - t1
    
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


class Point:
    '''
    Each point have 2 main values: coordinate(lat, long) and cluster_id
    '''
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, id:{}, cluster:{})' \
            .format(self.x, self.y, self.id, self.cluster_id)

# In G-DBScan we use elip instead of circle to cluster (because we mainly use for horizontal text image --> elip is more useful)
def n_pred_high(p1, p2):
    return (p1.x - p2.x)**2/6965 + (p1.y - p2.y)**2/145 <= 1
    # return (p1.x - p2.x)**2/6765 + (p1.y - p2.y)**2/140 <= 1

def n_pred_table(p1, p2):
    return (p1.x - p2.x)**2/3300 + (p1.y - p2.y)**2/85 <= 1


def n_pred_table_low(p1, p2):
    return (p1.x - p2.x)**2/1450 + (p1.y - p2.y)**2/70 <= 1

def n_pred_crop(p1, p2):
    return (p1.x - p2.x)**2/1406 + (p1.y - p2.y)**2/29 <= 1

def n_pred(p1, p2):
    return (p1.x - p2.x)**2/2250 + (p1.y - p2.y)**2/45 <= 1

def w_card(points):
    return len(points)


UNCLASSIFIED = -2
NOISE = -1
counter = 0


def GDBSCAN(points, n_pred, min_card, w_card):
    points = copy.deepcopy(points)
    cluster_id = 0
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if _expand_cluster(points, point, cluster_id, n_pred, min_card,
                               w_card):
                cluster_id = cluster_id + 1
    clusters = {}
    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]
    return list(clusters.values())


def _expand_cluster(points, point, cluster_id, n_pred, min_card, w_card):
    if not _in_selection(w_card, point):
        points.change_cluster_id(point, UNCLASSIFIED)
        return False

    seeds = points.neighborhood(point, n_pred)
    if not _core_point(w_card, min_card, seeds):
        points.change_cluster_id(point, NOISE)
        return False

    points.change_cluster_ids(seeds, cluster_id)
    seeds.remove(point)

    while len(seeds) > 0:
        current_point = seeds[0]
        result = points.neighborhood(current_point, n_pred)
        if w_card(result) >= min_card:
            for p in result:
                if w_card([p]) > 0 and p.cluster_id in [UNCLASSIFIED, NOISE]:
                    if p.cluster_id == UNCLASSIFIED:
                        seeds.append(p)
                    points.change_cluster_id(p, cluster_id)
        seeds.remove(current_point)
    return True


def _in_selection(w_card, point):
    return w_card([point]) > 0


def _core_point(w_card, min_card, points):
    return w_card(points) >= min_card


class Points:
    'Contain list of Point'

    def __init__(self, points):
        self.points = points

    def __iter__(self):
        for point in self.points:
            yield point

    def __repr__(self):
        return str(self.points)

    def get(self, index):
        return self.points[index]

    def neighborhood(self, point, n_pred):
        return list(filter(lambda x: n_pred(point, x), self.points))

    def change_cluster_ids(self, points, value):
        for point in points:
            self.change_cluster_id(point, value)

    def change_cluster_id(self, point, value):
        index = (self.points).index(point)
        self.points[index].cluster_id = value

    def labels(self):
        return set(map(lambda x: x.cluster_id, self.points))


trained_model_path = './craft_mlt_25k.pth'
net = CRAFT()
if torch.cuda.is_available():
    print('cuda')
    net.load_state_dict(copyStateDict(torch.load(
        trained_model_path, map_location='cuda:0')))
    net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
else:
    net.load_state_dict(copyStateDict(torch.load(
        trained_model_path, map_location='cpu')))
net.eval()


def seq2seq_vietocr(cnn_pretrained=False, beam_search=False, device='cpu'):
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['weights'] = './weights/transformerocr-2.pth'
    config['cnn']['pretrained'] = cnn_pretrained
    config['device'] = device
    config['predictor']['beamsearch'] = beam_search
    return Predictor(config)


def img_to_text_vietocr(image, isTable=False, isCrop=False):
    text_threshold = 0.455
    low_text = 0.399
    link_threshold = 0.445
    cuda = torch.cuda.is_available()
    canvas_size = 1280
    mag_ratio = 1.5
    poly = False

    refine_net = None
    bboxes, polys, score_text = test_net(
        canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)
    poly_indexes = {}
    central_poly_indexes = []

    for i in range(len(polys)):
        poly_indexes[i] = polys[i]
        x_central = (polys[i][0][0] + polys[i][1][0] +
                     polys[i][2][0] + polys[i][3][0])/4
        y_central = (polys[i][0][1] + polys[i][1][1] +
                     polys[i][2][1] + polys[i][3][1])/4
        central_poly_indexes.append({i: [int(x_central), int(y_central)]})
    X = []
    for idx, x in enumerate(central_poly_indexes):
        point = Point(x[idx][0], x[idx][1], idx)
        X.append(point)
    # file_utils.saveResult("./pre_newimage.jpg",
    #                       image[:, :, ::-1], polys, dirname='./imageStorage')
    poly = False
    refine_net = None
    if isTable:
        if image.shape[1] > 400:
            clustered = GDBSCAN(Points(X), n_pred_table, 1, w_card)
        else:
            clustered = GDBSCAN(Points(X), n_pred_table_low, 1, w_card)

    elif image.shape[1] < 800 and image.shape[1] < 1280:
        print('Low dimension image')
        clustered = GDBSCAN(Points(X), n_pred, 1, w_card)
    elif image.shape[1] < 800:
        clustered = GDBSCAN(Points(X), n_pred_crop, 1, w_card)

    elif isCrop and image.shape[1] > 800:
        print('High dimension image')
        clustered = GDBSCAN(Points(X), n_pred_high, 1, w_card)
    else:
        clustered = GDBSCAN(Points(X), n_pred, 1, w_card)
    cluster_values = []
    for cluster in clustered:
        sort_cluster = sorted(cluster, key=lambda elem: (elem.x, elem.y))
        max_point_id = sort_cluster[len(sort_cluster) - 1].id
        min_point_id = sort_cluster[0].id
        max_rectangle = sorted(
            poly_indexes[max_point_id], key=lambda elem: (elem[0], elem[1]))
        min_rectangle = sorted(
            poly_indexes[min_point_id], key=lambda elem: (elem[0], elem[1]))

        right_above_max_vertex = max_rectangle[len(max_rectangle) - 1]
        right_below_max_vertex = max_rectangle[len(max_rectangle) - 2]
        left_above_min_vertex = min_rectangle[0]
        left_below_min_vertex = min_rectangle[1]

        if (int(min_rectangle[0][1]) > int(min_rectangle[1][1])):
            left_above_min_vertex = min_rectangle[1]
            left_below_min_vertex = min_rectangle[0]
        if (int(max_rectangle[len(max_rectangle) - 1][1]) < int(max_rectangle[len(max_rectangle) - 2][1])):
            right_above_max_vertex = max_rectangle[len(max_rectangle) - 2]
            right_below_max_vertex = max_rectangle[len(max_rectangle) - 1]

        cluster_values.append([left_above_min_vertex, left_below_min_vertex,
                              right_above_max_vertex, right_below_max_vertex])
    # file_utils.saveResult('./dbscan_pre_newimage.jpg',
    #                       image[:, :, ::-1], cluster_values, dirname='imageStorage')

    img = np.array(image[:, :, ::-1])
    res = []

    if torch.cuda.is_available():
        ocr = seq2seq_vietocr(device='cuda:0')
    else:
        ocr = seq2seq_vietocr()

    for i, box in enumerate(cluster_values):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        poly[0][1] = poly[0][1]-1
        poly[1][1] = poly[1][1]+1
        poly[2][1] = poly[2][1]+1
        poly[3][1] = poly[3][1]-1
        rect = cv2.boundingRect(poly)
        x, y, w, h = rect

        croped = img[y:y+h, x:x+w].copy()
        pts = poly
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        # (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst
        if croped.shape[0] == 0 or croped.shape[1] == 0:
            continue
        ''' Condition for process '''

        img_croped = Image.fromarray(dst2)
        text = ocr.predict(img_croped)

        res.append(text)
    return res
