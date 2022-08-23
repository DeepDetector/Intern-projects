import numpy as np
import xml.etree.ElementTree as ET
import cv2
import pdb


def area(coord):  # get bbox area
    [l, t, r, b] = coord
    if l >= r or t >= b:
        return 0
    else:
        return (r - l + 1) * (b - t + 1)

def eval_recall_at_precision(
        all_images,
        all_gt_boxes,
        all_gt_labels,
        all_results,
        all_scores,
        prec_val,
        mul,
        temp,
        ovthresh=0.5,
        small_obj=False,
        all_gt_ignores=[]):
    # output threshold, recall
    for thd in np.arange(0.0 + temp, 0.999, 0.001 * mul):
    # for thd in np.arange(0.65, 1, 0.0001):
        prec_thd, rec_thd, det_gt_maps = get_prec_rec_at_thd(
            all_images,
            all_gt_boxes,
            all_gt_labels,
            all_results,
            all_scores,
            thd,
            ovthresh,
            with_ignore=small_obj,
            all_gt_ignores=all_gt_ignores)
        print("Now thd is {:.4f}".format(thd), "Prec_thd is {:.4f}".format(prec_thd), "max Recall is {:.4f}".format(rec_thd))
        if prec_thd > prec_val:
            break
    return thd, prec_thd, rec_thd, det_gt_maps


def get_ovmax_argmax(box, gts, with_ignore, gt_ignores_):
    ovmax, ovmax_ignore, argmax, argmax_ignore = -1, -1, -1, -1
    eps = 1e-3

    gts_np = np.array(gts).reshape(-1, 4).astype(int)
    lt = np.maximum(box[:2].astype(int), gts_np[:, :2])
    rb = np.minimum(box[2:].astype(int), gts_np[:, 2:])
    wh = (rb - lt + 1).clip(0)
    overlap = wh[:, 0] * wh[:, 1]
    gt_areas = (gts_np[:, 2] - gts_np[:, 0] + 1) * (
        gts_np[:, 3] - gts_np[:, 1] + 1)
    ious = overlap / np.maximum(gt_areas + area(box) - overlap, eps)

    overlap_ration = overlap / np.maximum(area(box), eps)

    for ip in range(len(overlap_ration)):
        if with_ignore and gt_ignores_[ip] == 1:
            if overlap_ration[ip] > ovmax_ignore:
                ovmax_ignore = overlap_ration[ip]
                argmax_ignore = ip
        else:
            if ious[ip] > ovmax:
                ovmax = ious[ip]
                argmax = ip
    return ovmax, argmax, ovmax_ignore, argmax_ignore


def get_prec_rec_at_thd(
        all_images,
        all_gt_boxes,
        all_gt_labels,
        all_results,
        all_scores,
        thd,
        ovthresh=0.5,
        with_ignore=False,
        all_gt_ignores=[]):
    if all_gt_boxes.count([]) == len(all_gt_boxes):
        return 0., 0., None
    tp_thd = 0.0
    fp_thd = 0.0
    npos_thd = 0.0
    eps = 1e-3  # prevent float division by zero
    image_num = len(all_gt_boxes)
    num_classes = len(all_results[0])
    
    det_gt_maps = []
    import time
    for cur_cls in range(0, num_classes):  # Note: i start from 1
        results = []  # list of all det bboxes
        scores = []
        belong_to = []
        det_gt_maps.append([])
        # prepare for calculating cur_cls's tp & fp
        _start_time = time.time()
        for j in range(image_num):
            for k in range(len(all_results[j][cur_cls])):
                if all_scores[j][cur_cls][k] > thd:
                    belong_to.append(j)
                    results.append(all_results[j][cur_cls][k])
                    scores.append((all_scores[j][cur_cls][k], len(scores)))
                    det_gt_maps[-1].append([j, cur_cls, k])
        # sort scores
        # pdb.set_trace()
        scores.sort()
        scores.reverse()  # greater: larger->small
        _end_time = time.time()
        #print('part 1 cost time {:.4f}s'.format(_end_time - _start_time))

        gt_boxes = [[] for each in range(image_num)]
        detected = [[] for each in range(image_num)]
        gt_ignores = [[] for each in range(image_num)]  # gt ignored

        # pdb.set_trace()

        _start_time = time.time()
        for j in range(image_num):
            for l in range(len(all_gt_labels[j])):
                if all_gt_labels[j][l] == cur_cls + 1:
                    gt_boxes[j].append(all_gt_boxes[j][l])
                    detected[j].append(False)
                    if with_ignore:
                        gt_ignores[j].append(all_gt_ignores[j][l])
                        if all_gt_ignores[j][l] == 0: # 1 is ignore
                            npos_thd += 1.0
                    else:
                        npos_thd += 1.0
                elif all_gt_labels[j][l] == 0:
                    if with_ignore:
                        gt_boxes[j].append(all_gt_boxes[j][l])
                        detected[j].append(False)
                        assert all_gt_ignores[j][l] == 1
                        gt_ignores[j].append(all_gt_ignores[j][l])
        _end_time = time.time()
        #print('part 2 cost time {:.4f}s'.format(_end_time - _start_time))
        # calculate fp, tp
        # pdb.set_trace()
        _start_time = time.time()
        assert len(belong_to) == len(scores)

        for j in range(len(belong_to)):
            result_index = scores[j][1]
            image_index = belong_to[result_index]

            ovmax, argmax, ovmax_ignore, _ = get_ovmax_argmax(
                results[result_index],
                gt_boxes[image_index],
                with_ignore,
                gt_ignores[image_index])

            visual_image = cv2.imread(all_images[image_index])
            for ig, b in zip(gt_ignores[image_index], gt_boxes[image_index]):
                if ig == 0:
                    visual_image = cv2.rectangle(visual_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,255,0), 1)
                else:
                    visual_image = cv2.rectangle(visual_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,0,0), 1)


            if (ovmax >= ovthresh) and (not detected[image_index][argmax]):
                detected[image_index][argmax] = True
                tp_thd += 1
                det_gt_maps[-1][result_index].append(argmax)  
            else:
                if ovmax_ignore < ovthresh:
                    fp_thd += 1
                    
                    visual_image = cv2.rectangle(visual_image, (int(results[result_index][0]), int(results[result_index][1])), (int(results[result_index][2]), int(results[result_index][3])), (0,0,255), 1)
                    cv2.imwrite(f'./visual/{image_index}.jpg', visual_image)
                    # cv2.imshow('fasdf', visual_image)
                    # cv2.waitKey()  
                    # cv2.destroyAllWindows()   
        # _end_time = time.time()
        # print(_end_time)
        #print('part 3 cost time {:.4f}s'.format(_end_time - _start_time))

    prec_thd = tp_thd / max(tp_thd + fp_thd, float(eps))
    rec_thd = tp_thd / max(npos_thd, float(eps))

    return prec_thd, rec_thd, det_gt_maps


def collect_result_message(all_gt_boxes,
                           all_gt_labels,
                           all_results,
                           all_scores,
                           thd,
                           ovthresh=0.5,
                           with_ignore=False,
                           all_gt_ignores=[]):
    eps = 1e-3  # prevent float division by zero
    image_num = len(all_gt_boxes)
    num_classes = len(all_results[0])
    result_attr = [[] for each in range(image_num)]  # 1->right 2->wrong
    for m_id, class_res in enumerate(all_results):
        result_attr[m_id] = [[] for each in range(len(class_res))]
        for c_id, num_res in enumerate(class_res):
            result_attr[m_id][c_id] = [0 for each in range(len(num_res))]
    detected = [[] for each in range(image_num)]
    gt_ignores = [[] for each in range(image_num)]  # gt ignored
    for j in range(image_num):
        for l in range(len(all_gt_labels[j])):
            detected[j].append(False)
            if with_ignore:
                gt_ignores[j].append(all_gt_ignores[j][l])
    for i in range(0, num_classes):  # Note: i start from 1
        cur_cls = i
        results = []  # list of all det bboxes
        scores = []
        belong_to = []
        # prepare for calculating cur_cls's tp & fp
        for j in range(image_num):
            for k in range(len(all_results[j][cur_cls])):
                if all_scores[j][cur_cls][k] > thd:
                    belong_to.append((j, cur_cls, k))
                    results.append(all_results[j][cur_cls][k])
                    scores.append((all_scores[j][cur_cls][k], len(scores)))
                else:
                    result_attr[j][cur_cls][k] = 2  # skip the low score box
        # sort scores
        scores.sort()
        scores.reverse()  # greater: larger->small

        # pdb.set_trace()

        gt_boxes = [[] for each in range(image_num)]
        for j in range(image_num):
            for l in range(len(all_gt_labels[j])):
                if all_gt_labels[j][l] == cur_cls + 1:
                    gt_boxes[j].append((all_gt_boxes[j][l], l))
        # calculate fp, tp
        for j in range(len(belong_to)):
            result_index = scores[j][1]
            image_index = belong_to[result_index][0]
            gts = [gt_item[0] for gt_item in gt_boxes[image_index]]
            ovmax, argmax, ovmax_ignore, _ = get_ovmax_argmax(
                results[result_index],
                gts,
                with_ignore,
                gt_ignores[image_index])

            gts = gt_boxes[image_index]
            if (ovmax >= ovthresh) and (not detected[image_index][gts[argmax][1]]):
                detected[image_index][gts[argmax][1]] = True
                result_attr[image_index][cur_cls][belong_to[result_index][2]] = 1  # right box
            else:
                if ovmax_ignore >= ovthresh:
                    result_attr[image_index][cur_cls][belong_to[result_index][2]] = 3  # for small obj

    lost_gt = []
    for j in range(image_num):
        for k, det in enumerate(detected[j]):
            if not det:
                if with_ignore:
                    if gt_ignores[j][k] == 0:
                        lost_gt.append((j, k))
                else:
                    lost_gt.append((j, k))
    return result_attr, lost_gt

def eval_class_precison_and_recall(det_gt_maps,
                                  all_extra_cls,
                                  gt_extra_cls,
                                  index):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    true_hand, false_hand = [], []
    gt_hand_up, gt_hand_down = [], []

    eps = 1e-3
    for i in range(len(det_gt_maps[index][0])):
        if len(det_gt_maps[index][0][i]) == 3:  # det box is neg/ignore
            continue
        img_ind, cls_ind, det_ind, gt_ind = det_gt_maps[index][0][i]
        det_label, det_score_0, det_score_1 = all_extra_cls[index][img_ind][det_ind]
        det_score = det_score_1
        gt_label = gt_extra_cls[index][img_ind][gt_ind]
        if int(gt_label) == 2:
            continue
        if int(gt_label) == 1:  # TP + FN
            gt_hand_up.append(1)
            true_hand.append(det_score)
        elif int(gt_label) == 0:  # int(gt_handup_cur[argmax]) != 1: # TN + FP
            gt_hand_down.append(0)
            false_hand.append(det_score)
    final_acc = 0.00
    hand_thd_50 = True
    hand_thd_85 = True
    fpr_thd_10 = True
    fpr_thd_1 = True
    PR_AUC, FPR_TPR_AUC = 0, 0
    TPR_FPR_10, TPR_FPR_1 = 0, 0
    FPR_FPR_10, FPR_FPR_1 = 0, 0
    Last_pre, Last_rec = 0, 0
    Last_pre_status, Last_rec_status = True, True
    prec_res_hand, rec_res_hand = 0, 0
    for hand_thd in np.arange(0.00, 1.00, 0.005):
        tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
        for i in range(len(gt_hand_up)):
            if(true_hand[i] >= hand_thd): #predict is true
                tp += 1
            else: # predict is false
                fn += 1
        for i in range(len(gt_hand_down)):
            if(false_hand[i] >= hand_thd): # predict is true
                fp += 1
            else:   #predict is false
                tn += 1

        prec_hand = tp / max(tp + fp, float(eps))
        rec_hand = tp / max(tp + fn, float(eps))
        f1_score = (2 * prec_hand * rec_hand)/max(prec_hand + rec_hand, float(eps))
        acc_hand = (tp + tn) / max((tp + fp + tn + fn), float(eps))
        TPR = tp / max(tp + fn, float(eps))
        FPR = fp / max(tn + fp, float(eps))
        # print("Now hand_thd is {:.4f}"
        #       "TPR is {:.4f} FPR is {:.4f} "
        #       "accuracy is {:.4f}, F1_score is{:.4f} "
        #       # "Recall_pos_hand is {:.4f} Recall_neg_hand is {:.4f} "
        #       .format(hand_thd,
        #               TPR, FPR,
        #               acc_hand, f1_score))
        if Last_rec_status:
            Last_TPR = TPR
            Last_FPR = FPR
            Last_pre = prec_hand
            Last_rec = rec_hand
            Last_rec_status = False
        else:
            FPR_TPR_area = ((Last_FPR - FPR ) * (Last_TPR + TPR) / 2)
            PR_AREA = (prec_hand - Last_pre) * (rec_hand + Last_rec) / 2
            FPR_TPR_AUC += FPR_TPR_area
            PR_AUC += PR_AREA
            Last_TPR = TPR
            Last_FPR = FPR
            Last_pre = prec_hand
            Last_rec = rec_hand
        if FPR < 0.1 and fpr_thd_10:
            TPR_FPR_10 = TPR
            FPR_FPR_10 = FPR
            fpr_thd_10 = False
            final_acc = acc_hand
        elif FPR < 0.01 and fpr_thd_1:
            FPR_FPR_1 = FPR
            TPR_FPR_1 = TPR
            fpr_thd_1 = False
        if prec_hand > 0.5 and hand_thd_50:
            prec_res_hand = prec_hand
            rec_res_hand = rec_hand
            hand_thd_50 = False
        elif prec_hand > 0.85 and hand_thd_85:
            #prec_res_hand = prec_hand
            #rec_res_hand = rec_hand
            hand_thd_85 = False
    #all_sum = 0.0
    #for img_ind in range(len(gt_extra_cls[ind])):
    #    assert -1 not in gt_extra_cls[ind][img_ind]
    #    all_sum += len(gt_extra_cls[ind][img_ind])
    # 90.71 17
    #     print('[Classification]'
    #           'Thd {:.4f} Precision {:.4f} Recall {:.4f} Accuracy {:.4f}'.format(
    #         hand_thd,
    #         prec_hand,
    #         rec_hand,
    #         acc_hand))
    print('[Classification]'
          'Thd {:.4f} prec_res_hand {:.4f} rec_res_hand {:.4f} Accuracy {:.4f}'.format(
        hand_thd,
        prec_res_hand,
        rec_res_hand,
        final_acc))
    print('[Classification]'
          'TPR_FPR_10 {:.4f} FPR_FPR_10 {:.4f} PR_AUC {:.4f} FPR_TPR_AUC {:.4f}'.format(
        TPR_FPR_10,
        FPR_FPR_10,
        PR_AUC,
        FPR_TPR_AUC)
    )
    return hand_thd, TPR_FPR_10, FPR_FPR_10, PR_AUC, FPR_TPR_AUC

def eval_pair_precision_and_recall(gt_corpairs,
                                   cor_results,
                                   iou_thr=0.5,
                                   thd=[],
                                   with_ignore=False,
                                   ignore_boxes=None):
    total_pos = 0
    total_pred = 0
    total_gt = 0
    assert len(cor_results) == len(gt_corpairs)
    for gt_pair, res_pair, ignore_box in zip(
            gt_corpairs, cor_results, ignore_boxes):
        gt_pair_num = gt_pair.shape[0]
        res_pair_num = res_pair.shape[0]
        detected = [False for _ in range(gt_pair_num)]
        total_gt += gt_pair_num
        total_pred += res_pair_num

        if gt_pair.size == 0:
            continue
        is_ignore = [0 for _ in range(gt_pair.shape[0])] + [
            1 for _ in range(ignore_box.shape[0])]
        if gt_pair.size > 0:
            all_boxes = np.concatenate(
                [gt_pair[:, :4], ignore_box.reshape(-1, 4)])
        else:
            all_boxes = ignore_box
        for res_i in range(res_pair_num):
            # 1. ignore box with score < thd
            if len(thd) > 0:
                assert len(thd) == 2
                if res_pair[res_i, 5] < thd[0]:
                    total_pred -= 1
                    continue
                if res_pair[res_i, -1] < thd[1]:
                    total_pred -= 1
                    continue
            # 2. check cor box is ignore
            #res_cor_box = res_pair[res_i, 4:]
            res_cor_box = res_pair[res_i, 5:9]
            _, _, ovmax_ignore, _ = get_ovmax_argmax(
                res_cor_box,
                ignore_box,
                with_ignore,
                [1 for _ in range(ignore_box.shape[0])])
            if ovmax_ignore >= iou_thr:
                total_pred -= 1
                continue
            # 3. match start box
            ovmax, argmax, ovmax_ignore, _ = get_ovmax_argmax(
                res_pair[res_i, :4], all_boxes, with_ignore, is_ignore)
            if ovmax >= iou_thr and not detected[argmax]:  # found gt start box
                detected[argmax] = True
                # then compute cor box's iou wiht gt box
                #res_cor_box = res_pair[res_i, 4:].astype(int)
                res_cor_box = res_pair[res_i, 5:9].astype(int)
                gt_cor_box = gt_pair[argmax, 4:].astype(int)
                ovbox = [max(gt_cor_box[0], res_cor_box[0]),
                         max(gt_cor_box[1], res_cor_box[1]),
                         min(gt_cor_box[2], res_cor_box[2]),
                         min(gt_cor_box[3], res_cor_box[3])]
                inter = area(ovbox)
                suma = max(area(res_cor_box) + area(gt_cor_box) - inter, 1e-3)
                overlap = inter * 1.0 / suma
                if overlap >= iou_thr:
                    total_pos += 1
            else:
                if ovmax_ignore >= iou_thr:  # start box is ignore box
                    total_pred -= 1
    precision = 1.0 * total_pos / max(total_pred, 1e-3)
    recall = 1.0 * total_pos / total_gt if total_gt != 0 else 0
    return precision, recall



def get_boxes_and_scores(annot_path):
    lines = open(annot_path).read().splitlines()

    bboxes = []
    scores = []
    labels = []

    for line in lines:
        if 'gts' not in annot_path:
            label, score, xmin, ymin, xmax, ymax = line.split(' ')
            bboxes.append(np.array((float(xmin), float(ymin), float(xmax), float(ymax))))
            scores.append(float(score))
            labels.append(1)
        else:
            label, xmin, ymin, xmax, ymax = line.split(' ')
            bboxes.append(np.array((float(xmin), float(ymin), float(xmax), float(ymax))))
            labels.append(1)

    return bboxes, labels, scores
def get_voc_annot(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult')
        difficult = 0 if difficult is None else int(difficult.text)
        bnd_box = obj.find('bndbox')
        # TODO: check whether it is necessary to use int
        # Coordinates may be float type
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(1)
        else:
            bboxes.append(bbox)
            labels.append(1)
    # if not bboxes_ignore:
    #     bboxes_ignore = np.zeros((0, 4))
    #     labels_ignore = np.zeros((0, ))
    # else:
    #     bboxes = np.array(bboxes, ndmin=2) - 1
    #     labels = np.array(labels)
    # if not bboxes_ignore:
    #     bboxes_ignore = np.zeros((0, 4))
    #     labels_ignore = np.zeros((0, ))
    # else:
    #     bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
    #     labels_ignore = np.array(labels_ignore)
    # ann = dict(
    #     bboxes=bboxes.astype(np.float32),
    #     labels=labels.astype(np.int64),
    #     bboxes_ignore=bboxes_ignore.astype(np.float32),
    #     labels_ignore=labels_ignore.astype(np.int64))
    return bboxes, labels, bboxes_ignore, labels_ignore

if __name__ == '__main__':
    # eval_recall_at_precision(all_gt_boxes,
    #                          all_gt_labels,
    #                          all_results,
    #                          all_scores,
    #                          prec_val,
    #                          mul,
    #                          temp,
    #                          ovthresh=0.5,
    #                          small_obj=False,
    #                          all_gt_ignores=[]):

    from glob import glob 
    import os
    
    all_images = sorted(glob('/raid/AI_lai/CYJ/data/sku_detection/20211207_online_test/val/JPEGImages_match/*.jpg'))
    gt_root = '/raid/AI_lai/CYJ/data/sku_detection/20211207_online_test/val/Annotations/'
    # pred_root = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/runs/detect/env/labels/'
    pred_root = '/raid/AI_lai/share/wb/code/yolov7-main/runs/detect/env/labels/'

    # test_ds_types = [x for x in os.listdir(gt_root) if x != 'none']

    def get_gt_boxes_and_scores(annot_path):
        lines = open(annot_path).read().splitlines()

        bboxes = []
        scores = []
        labels = []

        for line in lines:
            label, xmin, ymin, xmax, ymax = line.split(' ')
            bboxes.append(np.array((float(xmin), float(ymin), float(xmax), float(ymax))))
            labels.append(1)

        return bboxes, labels, scores

    def get_pred_boxes_and_scores(annot_path):
        lines = open(annot_path).read().splitlines()

        bboxes = []
        scores = []
        labels = []

        for line in lines:
            label, score, xmin, ymin, xmax, ymax = line.split(' ')
            bboxes.append(np.array((float(xmin), float(ymin), float(xmax), float(ymax))))
            scores.append(float(score))
            labels.append(1)

        return bboxes, labels, scores


    all_gt_boxes, all_gt_labels, all_gt_ignores = [], [], []
    all_results, all_labels, all_scores = [], [], []

    # all_results = [           pic
    #   [                       class
    #       []                  bboxlist
    #   ]
    # 
    # 
    # ]

    for index, (gt_annot, pre_annot) in enumerate(zip(sorted(glob(gt_root + '/*')), sorted(glob(pred_root + '/*')))):
        assert gt_annot.split('/')[-1][:-4] == pre_annot.split('/')[-1][:-4], f"{gt_annot}, {pre_annot}"
        # temp_all_gt_boxes, temp_all_gt_labels, _ = get_boxes_and_scores(gt_annot)
        bboxes, labels, bboxes_ignore, labels_ignore = get_voc_annot(gt_annot)
        temp_all_gt_boxes = bboxes + bboxes_ignore
        temp_all_gt_labels = [1] * len(labels) + [0] * len(labels_ignore)
        temp_all_gt_ignores = [0]*len(bboxes) + [1]*len(bboxes_ignore)


        temp_all_results, temp_all_labels, temp_all_scores = get_boxes_and_scores(pre_annot)
        
        # all_gt_boxes.append([])
        # all_gt_labels.append([])
        all_results.append([])
        # all_labels.append([])
        all_scores.append([])

        all_gt_boxes.append(temp_all_gt_boxes)
        all_gt_labels.append(temp_all_gt_labels)
        all_gt_ignores.append(temp_all_gt_ignores)
        all_results[-1].append(temp_all_results)
        all_labels.append(temp_all_labels)
        all_scores[-1].append(temp_all_scores)

    # print(all_scores)
    thd, prec_thd, rec_thd, det_gt_maps = eval_recall_at_precision(all_images, all_gt_boxes, all_gt_labels, all_results, all_scores, 0.7, 10, 0.01, small_obj=True, all_gt_ignores=all_gt_ignores)
    print('Evaluation result:')
    print(thd, prec_thd, rec_thd)
