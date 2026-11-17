"""Pure numpy/torch port of ActionFormer's 1D (soft-)NMS (original: libs/utils/csrc/nms_cpu.cpp).

Removes the need to compile the C++ extension (painful on Windows / isolated graders). Same
semantics: greedy max selection; gaussian soft-NMS (method=2) multiplies remaining scores by
exp(-iou^2 / sigma) and drops those below min_score.
"""
import numpy as np
import torch


def _softnms_np(segs, scores, iou_threshold, sigma, min_score, method):
    x1, x2 = segs[:, 0].astype(np.float64), segs[:, 1].astype(np.float64)
    sc = scores.astype(np.float64).copy()
    areas = x2 - x1 + 1e-6
    alive = np.ones(len(sc), bool)
    keep_idx, keep_sc = [], []
    while alive.any():
        cand = np.where(alive)[0]
        i = cand[np.argmax(sc[cand])]
        keep_idx.append(i)
        keep_sc.append(sc[i])
        alive[i] = False
        rest = np.where(alive)[0]
        if not len(rest):
            break
        inter = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ovr = inter / (areas[i] + areas[rest] - inter)
        if method == 0:
            w = np.where(ovr >= iou_threshold, 0.0, 1.0)
        elif method == 1:
            w = np.where(ovr >= iou_threshold, 1 - ovr, 1.0)
        else:
            w = np.exp(-(ovr * ovr) / sigma)
        sc[rest] *= w
        alive[rest[sc[rest] < min_score]] = False
    return np.array(keep_idx, np.int64), np.array(keep_sc, np.float32)


def _class_nms(segs, scores, cls_idxs, iou_threshold, sigma, min_score, method, max_num):
    if method == 0 and min_score > 0:
        m = scores > min_score
        segs, scores, cls_idxs = segs[m], scores[m], cls_idxs[m]
    if segs.shape[0] == 0:
        return segs, scores, cls_idxs
    idx, new_sc = _softnms_np(segs.cpu().numpy(), scores.cpu().numpy(), iou_threshold, sigma,
                              min_score if method != 0 else -1.0, method)
    if max_num > 0:
        idx, new_sc = idx[:max_num], new_sc[:max_num]
    it = torch.from_numpy(idx)
    out_scores = torch.from_numpy(new_sc) if method != 0 else scores.cpu()[it]
    return segs.cpu()[it].clone(), out_scores.clone(), cls_idxs.cpu()[it].clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)
    left = torch.maximum(ex_nms[:, :, 0], ex_all[:, :, 0])
    right = torch.minimum(ex_nms[:, :, 1], ex_all[:, :, 1])
    inter = (right - left).clamp(min=0)
    iou = inter / ((ex_nms[:, :, 1] - ex_nms[:, :, 0]) + (ex_all[:, :, 1] - ex_all[:, :, 0]) - inter)
    w = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :] * iou
    w /= torch.sum(w, dim=1, keepdim=True)
    return w @ all_segs


def batched_nms(segs, scores, cls_idxs, iou_threshold, min_score, max_seg_num,
                use_soft_nms=True, multiclass=True, sigma=0.5, voting_thresh=0.75):
    if segs.shape[0] == 0:
        return torch.zeros([0, 2]), torch.zeros([0]), torch.zeros([0], dtype=cls_idxs.dtype)
    method = 2 if use_soft_nms else 0
    if multiclass:
        outs = [_class_nms(segs[cls_idxs == c], scores[cls_idxs == c], cls_idxs[cls_idxs == c],
                           iou_threshold, sigma, min_score, method, max_seg_num)
                for c in torch.unique(cls_idxs)]
        new_segs = torch.cat([o[0] for o in outs])
        new_scores = torch.cat([o[1] for o in outs])
        new_cls = torch.cat([o[2] for o in outs])
    else:
        new_segs, new_scores, new_cls = _class_nms(segs, scores, cls_idxs, iou_threshold, sigma,
                                                   min_score, method, max_seg_num)
        if voting_thresh > 0:
            new_segs = seg_voting(new_segs, segs.cpu(), scores.cpu(), voting_thresh)
    _, idxs = new_scores.sort(descending=True)
    k = min(max_seg_num, new_segs.shape[0])
    return new_segs[idxs[:k]], new_scores[idxs[:k]], new_cls[idxs[:k]]
