import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import argparse
import os

import pycocotools.mask as maskUtils
import torch

import openpyxl

from utils.bbox_overlaps import bbox_overlaps
import xml.etree.ElementTree as ET
import operator as op
import time

TABLE_HEAD = ["名称", "样本个数", "tp", "fp", "fn", "precision", "recall"]

test_img_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/JPEGImages"
test_xml_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/Annotations"
test_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/ImageSets/Main/test.txt"


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = float(box[0].text)
        y_min = float(box[1].text)
        x_max = float(box[2].text)
        y_max = float(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


def get_result(det):
    bboxes = []
    labels = []
    scores = []
    for *xyxy, conf, cls in det:
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        bboxes.append([x1, y1, x2, y2])
        labels.append(int(cls))
        scores.append(float(conf))

    return bboxes, labels, scores


def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half = opt.weights, opt.half

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)


    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Get names and colors
    names = load_classes(opt.names)
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    cls2id = dict(zip(names, range(0, len(names))))
    gt_cls_num = np.zeros((len(names)))
    tp = np.zeros((len(names)))
    fp = np.zeros((len(names)))
    fn = np.zeros((len(names)))
    tn = np.zeros((len(names)))
    acc = 0.0
    tot = 0.0
    with open(test_path, "r") as f:
        filenames = f.readlines()
        for filename in filenames:
            img_file = filename.strip() + ".jpg"
            xml_file = filename.strip() + ".xml"
            source = os.path.join(test_img_path, img_file)
            dataset = LoadImages(source, img_size=imgsz)
            xml_path = os.path.join(test_xml_path, xml_file)
            coords = read_xml(xml_path)
            if len(coords) is 0:
                print("No annotations\n")
                continue
            gt_bboxes = [coord[:4] for coord in coords]
            gt_labels = [coord[4] for coord in coords]
            for label in gt_labels:
                gt_cls_num[cls2id[label]] += 1
                tot += 1
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = torch_utils.time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = torch_utils.time_synchronized()

                # to float
                if half:
                    pred = pred.float()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                           multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for j, det in enumerate(pred):  # detections for image j
                    p, s, im0 = path, '', im0s

                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from imgsz to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        det_bboxes, det_labels, det_scores = get_result(det)
                        ious = bbox_overlaps(np.array(det_bboxes), np.array(gt_bboxes))
                        ious_max = ious.max(axis=1)
                        ious_argmax = ious.argmax(axis=1)
                        gt_matched_det = np.ones((len(gt_bboxes))) * -1
                        det_matched_gt = np.ones((len(det_bboxes))) * -1
                        gt_matched_scores = np.zeros((len(gt_bboxes)))
                        for i in range(0, len(det_bboxes)):
                            if ious_max[i] > 0.5:
                                target_gt = ious_argmax[i]
                                if gt_matched_scores[target_gt] < det_scores[i]:
                                    gt_matched_scores[target_gt] = det_scores[i]
                                    gt_matched_det[target_gt] = i
                                    det_matched_gt[i] = target_gt
                            else:
                                fp[det_labels[i]] += 1

                        for i in range(0, len(det_matched_gt)):
                            gt = int(det_matched_gt[i])
                            if gt > -1:
                                if op.eq(names[det_labels[i]], gt_labels[gt]):
                                    tp[det_labels[i]] += 1
                                    assert (tp[det_labels[i]] <= gt_cls_num[det_labels[i]])
                                    acc += 1
                                else:
                                    fp[det_labels[i]] += 1

    mat = np.zeros((len(names), len(TABLE_HEAD)))
    for i in range(0, len(names)):
        mat[i][0] = i
        mat[i][1] = gt_cls_num[i]
        mat[i][2] = tp[i]
        mat[i][3] = fp[i]
        mat[i][4] = fn[i]
        mat[i][5] = tp[i] / (tp[i] + fp[i])
        mat[i][6] = tp[i] / (tp[i] + fn[i])
        print("%s: %.0f gt, %.0f det, %.0f tp, precision: %.6f, recall: %.6f" %
              (names[i], gt_cls_num[i], tp[i] + fp[i], tp[i], tp[i] / (tp[i] + fp[i]), tp[i] / (tp[i] + fn[i])))

    if os.path.exists("rzx_statistics.xlsx"):
        os.remove("rzx_statistics.xlsx")
    workbook = openpyxl.Workbook("rzx_statistics.xlsx")
    sheet = workbook.create_sheet("sheet")
    sheet.append(TABLE_HEAD)
    for i in range(0, len(names)):
        label = names[i]
        sheet.append([label, "%.0f" % gt_cls_num[i], "%.0f" % tp[i], "%.0f" % fp[i], "%.0f" % fn[i],
                      "%.6f" % (tp[i] / (tp[i] + fp[i])), "%.6f" % (tp[i] / (tp[i] + fn[i]))])

    workbook.save("rzx_statistics.xlsx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
