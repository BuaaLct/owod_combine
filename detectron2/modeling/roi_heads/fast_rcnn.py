# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from operator import gt
from typing import Dict, Union
import torch
import os
import math
import shortuuid
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.store import Store
from detectron2.uno.nets import MultiHeadResNet
from detectron2.clip_utils import ClipProcess
from detectron2.modeling.binary_matcher import build_matcher
from detectron2.uno.nets import CosMLP
from detectron2.modeling.roi_heads.kl_loss import get_cluster_prob
import numpy as np

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]

logger = logging.getLogger(__name__)

def kl_div(A, B, mask=None, maskval=0., T=1):
    if mask is not None:
        log_p_A = F.log_softmax(A.masked_fill(mask, maskval) / T, dim=-1)
        p_B = F.softmax(B.masked_fill(mask, maskval) / T, dim=-1)
    else:
        log_p_A = F.log_softmax(A / T, dim=-1)
        p_B = F.softmax(B / T, dim=-1)
    kl_div = F.kl_div(log_p_A, p_B, reduction='sum') * (T ** 2) / A.shape[0]
    return kl_div
"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def uno_fast_rcnn_inference(uno_preds, boxes, scores, image_shapes, predictions, score_thresh, nms_thresh,
                            topk_per_image, seen_classes):
    result_per_image = [
        uno_fast_rcnn_inference_single_image(
            uno_pred, boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image,
            prediction, seen_classes
        )
        for uno_pred, scores_per_image, boxes_per_image, image_shape, prediction in
        zip(uno_preds, scores, boxes, image_shapes, predictions)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference(boxes, scores, image_shapes, predictions, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, prediction
        )
        for scores_per_image, boxes_per_image, image_shape, prediction in zip(scores, boxes, image_shapes, predictions)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def uno_fast_rcnn_inference_single_image_margin_1(
        uno_preds, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, prediction, seen_classes
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    matcher = build_matcher()
    logits = prediction
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        logits = logits[valid_mask]
        uno_preds = uno_preds[valid_mask]
    
    scores = scores[:, :-1]
    logits = logits[:, :-1]
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
    uno_scores=F.softmax(uno_preds,dim=-1)
    scores = scores[filter_mask]
    
    uno_scores=uno_scores[filter_inds[:, 0]]
    uno_logits = uno_preds[filter_inds[:, 0]]
    logits = logits[filter_inds[:, 0]]
    # print(logits.shape)[117, 81]
    # print(uno_scores.shape)[117, 30]
    # print(boxes.shape)[117, 4]
    # exit()
    origin_boxes = boxes.clone()
    # print(filter_inds[:, 1].shape)
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    origin_filter_inds = filter_inds.clone()

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    # t = torch.zeros(origin_boxes.shape[0])
    # t[keep] = 1
    # t_idx = t != 1
    if uno_scores.shape[0]!=0:
        uno_pred_idx=uno_scores.max(dim=-1)[1]

        uno_pred_idx[keep] = -1
        # uno_boxes = origin_boxes[t_idx]

        uno_idx = uno_pred_idx >= seen_classes
        uno_box = origin_boxes[uno_idx]
        uno_scores=uno_scores[uno_idx]
        if uno_box.shape[0]!=0:
            uno_scores = torch.max(uno_scores,dim=-1)[0]
            idxs=torch.ones(uno_scores.shape[0],device=uno_scores.device)*80
            indices=matcher(uno_scores,uno_box,boxes)[0]
            uno_mask=torch.tensor([item not in indices for item in torch.arange(uno_box.shape[0]) ])
            uno_box=uno_box[uno_mask]
            uno_scores=uno_scores[uno_mask]
            keep_uno = batched_nms(uno_box, uno_scores, idxs, nms_thresh)
            uno_box=uno_box[keep_uno]
            uno_scores=uno_scores[keep_uno]
            
            # print(boxes.shape,filter_inds[:, 1].shape)
            # print('after',uno_box.shape)
    else:
        uno_box=torch.zeros((0,4))

    logits = logits[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    # result.logits = logits

    if uno_box.shape[0]!=0:
        uno_result = Instances(image_shape)
        uno_result.pred_boxes = Boxes(uno_box)
        # print( torch.max(uno_scores,dim=-1)[0].shape,uno_box.shape)
        # # print(uno_scores.shape,logits[keep].shape,scores.shape)
        # exit()
        uno_result.scores=uno_scores
        # try:
        #     uno_result.scores = torch.max(uno_scores,dim=-1)[0]
        # except RuntimeError:
        #     print(uno_box.shape,uno_scores.shape)
        uno_result.pred_classes = torch.zeros(uno_scores.shape[0], device=scores.device) + 80
        # uno_result.logits = uno_logits[uno_idx]
        uno_filter_inds = origin_filter_inds[uno_idx]
        final_result = Instances.cat([result, uno_result])
        final_inds = torch.cat([filter_inds[:, 0], uno_filter_inds[:, 0]])
    else:
        final_result=result
        final_inds=filter_inds[:, 0]

    return final_result, final_inds

def uno_fast_rcnn_inference_single_image(
        uno_preds, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, prediction, seen_classes
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    matcher = build_matcher()
    logits = prediction
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        logits = logits[valid_mask]
        uno_preds = uno_preds[valid_mask]
    
    scores = scores[:, :-1]
    logits = logits[:, :-1]
    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
    uno_scores=F.softmax(uno_preds,dim=-1)
    scores = scores[filter_mask]
    
    uno_scores=uno_scores[filter_inds[:, 0]]
    uno_logits = uno_preds[filter_inds[:, 0]]
    logits = logits[filter_inds[:, 0]]
    # print(logits.shape)[117, 81]
    # print(uno_scores.shape)[117, 30]
    # print(boxes.shape)[117, 4]
    # exit()
    origin_boxes = boxes.clone()
    # print(filter_inds[:, 1].shape)
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    origin_filter_inds = filter_inds.clone()

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    # t = torch.zeros(origin_boxes.shape[0])
    # t[keep] = 1
    # t_idx = t != 1
    if uno_scores.shape[0]!=0:
        uno_pred_idx=uno_scores.max(dim=-1)[1]

        uno_pred_idx[keep] = -1
        # uno_boxes = origin_boxes[t_idx]

        uno_idx = uno_pred_idx >= seen_classes
        uno_box = origin_boxes[uno_idx]
        uno_scores=uno_scores[uno_idx]
        if uno_box.shape[0]!=0:
            uno_scores = torch.max(uno_scores,dim=-1)[0]
            idxs=torch.ones(uno_scores.shape[0],device=uno_scores.device)*80
            indices=matcher(uno_scores,uno_box,boxes)[0]
            uno_mask=torch.tensor([item not in indices for item in torch.arange(uno_box.shape[0]) ])
            uno_box=uno_box[uno_mask]
            uno_scores=uno_scores[uno_mask]
            keep_uno = batched_nms(uno_box, uno_scores, idxs, nms_thresh)
            uno_box=uno_box[keep_uno]
            uno_scores=uno_scores[keep_uno]
            
            # print(boxes.shape,filter_inds[:, 1].shape)
            # print('after',uno_box.shape)
    else:
        uno_box=torch.zeros((0,4))

    logits = logits[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    # result.logits = logits

    if uno_box.shape[0]!=0:
        uno_result = Instances(image_shape)
        uno_result.pred_boxes = Boxes(uno_box)
        # print( torch.max(uno_scores,dim=-1)[0].shape,uno_box.shape)
        # # print(uno_scores.shape,logits[keep].shape,scores.shape)
        # exit()
        uno_result.scores=uno_scores
        # try:
        #     uno_result.scores = torch.max(uno_scores,dim=-1)[0]
        # except RuntimeError:
        #     print(uno_box.shape,uno_scores.shape)
        uno_result.pred_classes = torch.zeros(uno_scores.shape[0], device=scores.device) + 80
        # uno_result.logits = uno_logits[uno_idx]
        uno_filter_inds = origin_filter_inds[uno_idx]
        final_result = Instances.cat([result, uno_result])
        final_inds = torch.cat([filter_inds[:, 0], uno_filter_inds[:, 0]])
    else:
        final_result=result
        final_inds=filter_inds[:, 0]

    return final_result, final_inds

# def uno_fast_rcnn_inference_single_image(
#         uno_preds, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, prediction, seen_classes
# ):
#     """
#     Single-image inference. Return bounding-box detection results by thresholding
#     on scores and applying non-maximum suppression (NMS).

#     Args:
#         Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
#         per image.

#     Returns:
#         Same as `fast_rcnn_inference`, but for only one image.
#     """
    
#     logits = prediction
#     valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
#     if not valid_mask.all():
#         boxes = boxes[valid_mask]
#         scores = scores[valid_mask]
#         logits = logits[valid_mask]
#         uno_preds = uno_preds[valid_mask]
    
#     scores = scores[:, :-1]
#     logits = logits[:, :-1]
    
#     num_bbox_reg_classes = boxes.shape[1] // 4
#     # Convert to Boxes to use the `clip` function ...
#     boxes = Boxes(boxes.reshape(-1, 4))
#     boxes.clip(image_shape)
#     boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
#     # 1. Filter results based on detection scores. It can make NMS more efficient
#     #    by filtering out low-confidence detections.
#     filter_mask = scores > score_thresh  # R x K

#     # R' x 2. First column contains indices of the R predictions;
#     # Second column contains indices of classes.
#     filter_inds = filter_mask.nonzero()
#     if num_bbox_reg_classes == 1:
#         boxes = boxes[filter_inds[:, 0], 0]
#     else:
#         boxes = boxes[filter_mask]
    
#     uno_scores=F.softmax(uno_preds,dim=-1)
#     scores = scores[filter_mask]
    
#     uno_scores=uno_scores[filter_inds[:, 0]]
#     uno_logits = uno_preds[filter_inds[:, 0]]
#     logits = logits[filter_inds[:, 0]]
#     # print(logits.shape)[117, 81]
#     # print(uno_scores.shape)[117, 30]
#     # print(boxes.shape)[117, 4]
#     # exit()
#     origin_boxes = boxes.clone()
    
#     # 2. Apply NMS for each class independently.
#     keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

#     origin_filter_inds = filter_inds.clone()

#     if topk_per_image >= 0:
#         keep = keep[:topk_per_image]
#     boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
#     # t = torch.zeros(origin_boxes.shape[0])
#     # t[keep] = 1
#     # t_idx = t != 1
#     if uno_scores.shape[0]!=0:
#         uno_pred_idx=uno_scores.max(dim=-1)[1]

#         uno_pred_idx[keep] = -1
#         # uno_boxes = origin_boxes[t_idx]

#         uno_idx = uno_pred_idx > seen_classes
#         uno_box = origin_boxes[uno_idx]
#         uno_scores = uno_scores[uno_idx]
#         print(uno_scores.shape,uno_box.shape)
#     else:
#         uno_box=torch.zeros((0,4))

#     logits = logits[keep]

#     result = Instances(image_shape)
#     result.pred_boxes = Boxes(boxes)
#     result.scores = scores
#     result.pred_classes = filter_inds[:, 1]
#     # result.logits = logits

#     if uno_box.shape[0]!=0:
#         uno_result = Instances(image_shape)
#         uno_result.pred_boxes = Boxes(uno_box)
#         try:
#             uno_result.scores = torch.max(uno_scores,dim=-1)[0]
#         except RuntimeError:
#             print(uno_box.shape,uno_scores.shape)
#         uno_result.pred_classes = torch.zeros(uno_scores.shape[0], device=scores.device) + 80
#         # uno_result.logits = uno_logits[uno_idx]
#         uno_filter_inds = origin_filter_inds[uno_idx]
#         final_result = Instances.cat([result, uno_result])
#         final_inds = torch.cat([filter_inds[:, 0], uno_filter_inds[:, 0]])
#     else:
#         final_result=result
#         final_inds=filter_inds[:, 0]

#     return final_result, final_inds

def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, prediction
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    logits = prediction
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        logits = logits[valid_mask]

    scores = scores[:, :-1]
    logits = logits[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    logits = logits[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    logits = logits[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.logits = logits
    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            sa_cls_scores,
            proposals,
            invalid_class_range,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.sa_cls_scores=sa_cls_scores

        self.image_shapes = [x.image_size for x in proposals]
        self.invalid_class_range = invalid_class_range

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            self.pred_class_logits[:, self.invalid_class_range] = -10e10
            # self.log_logits(self.pred_class_logits, self.gt_classes)
            # print(self.gt_classes.shape,self.pred_class_logits.shape)
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def log_logits(self, logits, cls):
        data = (logits, cls)
        location = '/home/fk1/workspace/OWOD/output/logits/' + shortuuid.uuid() + '.pkl'
        torch.save(data, location)

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), 
        "loss_box_reg": self.box_reg_loss(),
        "loss_sa_cls":self.sa_cross_entropy_loss()
        }

    def sa_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.sa_cls_scores.sum()
        else:
            self._log_accuracy()
            self.sa_cls_scores[:, self.invalid_class_range] = -10e10
            # self.log_logits(self.pred_class_logits, self.gt_classes)
            return F.cross_entropy(self.sa_cls_scores, self.gt_classes, reduction="mean")

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class AE(nn.Module):
    def __init__(self, input_size, z_dim):
        super(AE, self).__init__()
        self.e1 = nn.Linear(input_size, z_dim)
        self.d1 = nn.Linear(z_dim, input_size)

    def encoder(self, x):
        z = self.e1(x)
        z = torch.relu(z)
        return z

    def decoder(self, z):
        x = self.d1(z)
        x = torch.relu(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            clustering_items_per_class,
            clustering_start_iter,
            clustering_update_mu_iter,
            clustering_momentum,
            clustering_z_dimension,
            enable_clustering,
            prev_intro_cls,
            curr_intro_cls,
            max_iterations,
            output_dir,
            feat_store_path,
            margin,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        self.feature_to_sa = nn.Linear(2048, 512)
        self.sa_to_cls = nn.Linear(512, num_classes + 1)

        # attention module
        self.atten_cls_score = Linear(input_size*2 , num_classes + 1)
        self.Wy = nn.Linear(input_size, input_size)
        self.Wz = nn.Linear(512, input_size)
        nn.init.normal_(self.Wy.weight,std=0.02)
        nn.init.normal_(self.Wz.weight,std=0.02)
        nn.init.normal_(self.atten_cls_score.weight,std=0.01)
        for l in [self.Wz, self.Wy,self.atten_cls_score]:
            nn.init.constant_(l.bias, 0)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.feature_to_sa.weight,std=0.01)
        nn.init.normal_(self.sa_to_cls.weight,std=0.01)
        for l in [self.cls_score, self.bbox_pred,self.feature_to_sa,self.sa_to_cls]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.num_classes = num_classes
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.clustering_momentum = clustering_momentum

        self.hingeloss = nn.HingeEmbeddingLoss(2)
        self.enable_clustering = enable_clustering

        self.prev_intro_cls = prev_intro_cls
        self.curr_intro_cls = curr_intro_cls
        self.seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.invalid_class_range = list(range(self.seen_classes, self.num_classes - 1))
        logging.getLogger(__name__).info("Invalid class range: " + str(self.invalid_class_range))

        self.max_iterations = max_iterations
        self.feature_store_is_stored = False
        self.output_dir = output_dir
        self.feat_store_path = feat_store_path
        self.feature_store_save_loc = os.path.join(self.output_dir, self.feat_store_path, 'feat.pt')

        if os.path.isfile(self.feature_store_save_loc):
            logging.getLogger(__name__).info('Trying to load feature store from ' + self.feature_store_save_loc)
            self.feature_store = torch.load(self.feature_store_save_loc)
        else:
            logging.getLogger(__name__).info('Feature store not found in ' +
                                             self.feature_store_save_loc + '. Creating new feature store.')
            self.feature_store = Store(num_classes + 1, clustering_items_per_class)
        self.means = [None for _ in range(num_classes + 1)]
        self.margin = margin
        
        # 10 uno classes 
        self.uno_means = [None for _ in range(10)]
        self.uno_feature_store = Store(10, clustering_items_per_class)

        # self.ae_model = AE(input_size, clustering_z_dimension)
        # self.ae_model.apply(Xavier)

        self.uno_model = MultiHeadResNet(
            self.seen_classes,
            10,
        )

        sa_uno_model=MultiHeadResNet(self.seen_classes, 10, feat_dim=512)
        
        self.sa_unk_head_unlab=sa_uno_model.head_unlab

        self.clip_process = ClipProcess()

        self.mlp_model = CosMLP([2048, 512], hidden_dim = 1024, output_dim=1024, num_classes=(self.seen_classes + 11), 
        clip_process = self.clip_process, seen_classes=self.seen_classes)
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        self.update_start_iter = 16000
        
        self.language_shift = 1

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT, "loss_clustering": 0.1},
            "clustering_items_per_class": cfg.OWOD.CLUSTERING.ITEMS_PER_CLASS,
            "clustering_start_iter": cfg.OWOD.CLUSTERING.START_ITER,
            "clustering_update_mu_iter": cfg.OWOD.CLUSTERING.UPDATE_MU_ITER,
            "clustering_momentum": cfg.OWOD.CLUSTERING.MOMENTUM,
            "clustering_z_dimension": cfg.OWOD.CLUSTERING.Z_DIMENSION,
            "enable_clustering": cfg.OWOD.ENABLE_CLUSTERING,
            "prev_intro_cls": cfg.OWOD.PREV_INTRODUCED_CLS,
            "curr_intro_cls": cfg.OWOD.CUR_INTRODUCED_CLS,
            "max_iterations": cfg.SOLVER.MAX_ITER,
            "output_dir": cfg.OUTPUT_DIR,
            "feat_store_path": cfg.OWOD.FEATURE_STORE_SAVE_PATH,
            "margin": cfg.OWOD.CLUSTERING.MARGIN,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        # scores = self.uno_model.forward_scores(x)
        # with torch.no_grad():
        #     logits_unlab_max=outputs["logits_unlab"][0].max(dim=-1)[0][:,None]
        #     logits_lab=outputs["logits_lab"]
        #     scores=torch.cat((logits_lab[:,:-1],logits_unlab_max,logits_lab[:,-1:]),dim=-1)
        
        proposal_deltas = self.bbox_pred(x)
        sa_semantic = self.feature_to_sa(x)
        sa_cls_scores = self.sa_to_cls(sa_semantic)

        # scores = self.mlp_model.inference(x,sa_semantic,scores)
        return scores, proposal_deltas,sa_semantic,sa_cls_scores

    def update_feature_store(self, features, proposals):
        # cat(..., dim=0) concatenates over all images in the batch
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        self.feature_store.add(features, gt_classes)

        storage = get_event_storage()

        if storage.iter == self.max_iterations - 1 and self.feature_store_is_stored is False and comm.is_main_process():
            logging.getLogger(__name__).info(
                'Saving image store at iteration ' + str(storage.iter) + ' to ' + self.feature_store_save_loc)
            torch.save(self.feature_store, self.feature_store_save_loc)
            self.feature_store_is_stored = True

        # self.feature_store.add(F.normalize(features, dim=0), gt_classes)
        # self.feature_store.add(self.ae_model.encoder(features), gt_classes)

    def attentin_cls_score(self, input_features):
        device = input_features.device
        sa_dic = self.clip_process.get_text_features(device).cuda()
        
        projector_sa = self.Wz(sa_dic)
        # print('sa',sa_features.shape)
        attention = torch.mm(self.Wy(input_features), projector_sa.t()) / (self.input_size ** 0.5)
        attention[:,self.invalid_class_range] = 1e-10
        attention = F.softmax(attention, 1)
        
        z_hat = attention.unsqueeze(2) * projector_sa.unsqueeze(0)
        # print(z_hat.shape)
        z_hat = z_hat.sum(dim=1)
        # print(z_hat.shape)
        
        concat_features = torch.cat([input_features,z_hat],dim=1)
        # print(concat_features.shape)
        atten_scores = self.atten_cls_score(concat_features)
        # print(atten_scores.shape)
        return atten_scores
    
    def clstr_loss_l2_cdist(self, input_features, proposals):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        mask = gt_classes != self.num_classes
        fg_features = input_features[mask]
        classes = gt_classes[mask]
        # fg_features = F.normalize(fg_features, dim=0)
        # fg_features = self.ae_model.encoder(fg_features)

        all_means = self.means
        for item in all_means:
            if item != None:
                length = item.shape
                break

        for i, item in enumerate(all_means):
            if item == None:
                all_means[i] = torch.zeros((length))

        distances = torch.cdist(fg_features, torch.stack(all_means).cuda(), p=self.margin)
        labels = []

        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                if mu is not None and feature is not None:
                    if classes[index] == cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)

        loss = self.hingeloss(distances, torch.tensor(labels).reshape((-1, self.num_classes + 1)).cuda())

        return loss

    def get_uno_loss(self, feats, uno_classes, mask_lab,storage_all_feats,unknown_feats):
        return self.uno_model.get_uno_loss(feats, uno_classes, mask_lab,storage_all_feats,unknown_feats)


    def update_unk_classes(self, proposals, scores:torch.tensor):
        
        storage = get_event_storage()
        
        if storage.iter > self.update_start_iter :
            gt_classes = torch.cat([p.gt_classes for p in proposals])
            known_mask = gt_classes < self.seen_classes
            scores[:, self.invalid_class_range] = -10e10
            
            s_scores = F.softmax(scores,dim=-1)
            known_scores, class_ids = torch.max(s_scores[known_mask],dim=-1)
            
            valid_mask = class_ids < self.seen_classes
            valid_known_scores ,valid_class_ids = known_scores[valid_mask], class_ids[valid_mask]
            
            thresh_list = [np.empty((0)) for i in range(self.seen_classes)]
            for s,id in zip(valid_known_scores,valid_class_ids):
                thresh_list[id] = np.append(thresh_list[id],s.item())
            
            # caculate the thresh of each class if exists
            for i in range(self.seen_classes):
                thresh_list[i] = None if len(thresh_list[i]) == 0 else thresh_list[i].mean()
            
            num_prop_per_image = [len(p) for p in proposals]
            
            for proposal,proposal_scores in zip(proposals,s_scores.split(num_prop_per_image)):
                proposal_gt = proposal.gt_classes
                bg_mask = (proposal_gt >= self.seen_classes)
                bg_ids = bg_mask.nonzero().squeeze(-1)
                bg_scores,bg_classes = torch.max(proposal_scores[bg_mask],dim=-1)
                
                selected_mask = (bg_classes < self.seen_classes)
                selected_bg_ids,selected_bg_scores,selected_bg_classes =  bg_ids[selected_mask],bg_scores[selected_mask],bg_classes[selected_mask]
                
                for s_id,s_ss,s_class in zip(selected_bg_ids,selected_bg_scores,selected_bg_classes):
                    # situation 1 : s_ss smaller than thresh if thresh exists
                    if thresh_list[s_class] is not None and s_ss <= thresh_list[s_class]:
                        proposal.gt_classes[s_id] = self.num_classes-1
                        
                    # situation 2 : s_ss bigger than 0.05 if thresh not exists
                    elif thresh_list[s_class] is None and s_ss >= 0.05:
                        proposal.gt_classes[s_id] = self.num_classes-1
    
    def get_clustering_loss(self, input_features, proposals):
        if not self.enable_clustering:
            return 0

        storage = get_event_storage()
        c_loss = 0
        if storage.iter == self.clustering_start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    print(torch.tensor(item).shape)
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif storage.iter > self.clustering_start_iter:
            if storage.iter % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(self.num_classes + 1)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if (mean) is not None and new_means[i] is not None:
                        self.means[i] = self.clustering_momentum * mean + \
                                        (1 - self.clustering_momentum) * new_means[i]

            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return c_loss

    # def get_ae_loss(self, input_features):
    #     # storage = get_event_storage()
    #     # ae_loss = 0
    #     # if storage.iter < self.clustering_start_iter :
    #     features_hat = self.ae_model(input_features)
    #     ae_loss = F.mse_loss(features_hat, input_features)
    #     return ae_loss
    def get_sa_loss2(self, sa_semantic, proposals):
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        mask = gt_classes != self.num_classes
        fg_features = sa_semantic[mask]
        classes = gt_classes[mask]
        text_features = self.clip_process.get_text_features(sa_semantic.device)
        # print(text_features.shape)
        
        target = text_features[classes]
        mse = nn.MSELoss(reduction='mean')
        loss = mse(fg_features, target.to(fg_features.device))
        
        loss_margin=0
        # mask = (gt_classes != self.num_classes-1)
        fg_features = normalize(sa_semantic)
        # classes = gt_classes.clone()

        super_cls_list = self.clip_process.super_cls_list
        all_cls_set = set(i for i in range(self.seen_classes+2))
        index=torch.arange(self.seen_classes).tolist()
        index.append(self.num_classes-1)
        index.append(self.num_classes)
        fixed_matrix=normalize(text_features[index]).T.to(fg_features.device)
        fixed_scores=torch.matmul(fg_features,fixed_matrix)
        # print(fixed_scores)
        changed_gt=gt_classes.clone()
        changed_gt[changed_gt==self.num_classes-1]=self.seen_classes
        changed_gt[changed_gt==self.num_classes]=self.seen_classes+1
        # print(set(changed_gt.tolist()))
        for i in range(len(changed_gt)):
            sum_i=0
            for item in super_cls_list:
                
                if changed_gt[i].item() in item:
                    gt_set = item
                    # print(changed_gt[i],gt_set)
            remain_set = all_cls_set-gt_set

            for same_super_cls in gt_set:
                temp = torch.log(1+torch.exp(fixed_scores[i]-fixed_scores[i,same_super_cls]))
                sum_i += temp[list(remain_set)].sum()
            # fixed_scores[i] = torch.log(1+torch.exp(fixed_scores[i]-fixed_scores[i,changed_gt[i]]))
            
            # for j in range(self.seen_classes+1):
            #     if j!=changed_gt[i]:
            #         sum_i = sum_i+fixed_scores[i,j]
            # print(sum_i.shape)
            loss_margin += sum_i/(len(gt_set)*len(remain_set))
        
        return loss+loss_margin/classes.shape[0]

    def get_sa_loss3(self, sa_semantic, proposals,input_features):
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        mask = gt_classes != self.num_classes
        fg_features = sa_semantic[mask]
        classes = gt_classes[mask]
        text_features = self.clip_process.get_text_features(sa_semantic.device)
        # print(text_features.shape)
        
        target = text_features[classes]
        mse = nn.MSELoss(reduction='mean')
        loss = mse(fg_features, target.to(fg_features.device))

        unk_mask = gt_classes == self.num_classes-1
        unk_features = input_features[unk_mask]
        unk_sa_features = sa_semantic[unk_mask]
        logits_unlab = self.uno_model(unk_features)['logits_unlab'][0]
        pred_class = logits_unlab.max(dim=-1)[1]
        
        super_features = self.clip_process.get_super_features(sa_semantic.device)
        unk_targets = super_features[pred_class]
        loss2 = mse(unk_sa_features, unk_targets.to(fg_features.device))
        
        return loss+loss2
    
    def get_sa_loss(self, sa_semantic, proposals):
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        mask = gt_classes != self.num_classes
        fg_features = sa_semantic[mask]
        classes = gt_classes[mask]
        text_features = self.clip_process.get_text_features(sa_semantic.device)
        # print(text_features.shape)
        
        target = text_features[classes]
        mse = nn.MSELoss(reduction='mean')
        loss = mse(fg_features, target.to(fg_features.device))
        
        loss_margin=0
        mask = (gt_classes != self.num_classes-1)
        fg_features = normalize(sa_semantic[mask])
        classes = gt_classes[mask]


        index=torch.arange(self.seen_classes).tolist()
        index.append(self.num_classes)
        fixed_matrix=normalize(text_features[index]).T.to(fg_features.device)
        fixed_scores=torch.matmul(fg_features,fixed_matrix)
        # print(fixed_scores)
        changed_gt=classes.clone()
        changed_gt[changed_gt>self.seen_classes]=self.seen_classes
        for i in range(len(changed_gt)):
            sum_i=0
            fixed_scores[i] = torch.log(1+torch.exp(fixed_scores[i]-fixed_scores[i,changed_gt[i]]))
            
            for j in range(self.seen_classes+1):
                if j!=changed_gt[i]:
                    sum_i = sum_i+fixed_scores[i,j]
                
                # sum_i = sum_i+fixed_scores[i,j] if j!=changed_gt[i] else sum_i
            loss_margin += sum_i/self.seen_classes
        
        return loss+loss_margin/classes.shape[0]

    def get_unk_sa_loss(self,unknown):
        unk_samantic_features=self.feature_to_sa(unknown[0])
        unk_sa_scores=self.sa_unk_head_unlab(unk_samantic_features)[0][0]
        cls=unknown[1]
        unknwon_num=unknown[0].shape[0]
        unknwon_max_logits=torch.zeros((unknwon_num,unknwon_num),device=unknown[0].device)
        for i in range(unknwon_num):
            for j in range(unknwon_num):
                unknwon_max_logits[i,j]=unk_sa_scores[i,cls[i]]*unk_sa_scores[j,cls[j]]
        unknwon_max_logits=unknwon_max_logits.view(-1)
        unk_targets=(cls[None,:]==cls[:,None]).to(torch.float32).view(-1)
        # print(unk_targets)
        loss_unk_sa=F.binary_cross_entropy_with_logits(
                unknwon_max_logits,
                unk_targets,
                reduction="mean",
            )
        # print(loss_unk_sa)
        return loss_unk_sa*0.1
        # print(loss_unk_sa)

    def get_mlp_loss(self,sa_semantic,proposals, input_features):
        
        gt_classes = torch.cat([p.gt_classes for p in proposals])

        device = input_features.device
        sa_features = self.clip_process.get_text_features(device)
        known_sa_features = sa_features[:self.seen_classes]
        bg_sa_features = sa_features[-1:,:]
        unk_sa_features = self.clip_process.get_super_features(device)
        sa_concat_features = torch.cat([known_sa_features,unk_sa_features,bg_sa_features]).to(device)
        
        fg_mask = gt_classes != self.num_classes
        fg_features = input_features[fg_mask]

        unk_mask = gt_classes == self.num_classes-1
        unk_features = input_features[unk_mask]
        logits_unlab = self.uno_model(unk_features)['logits_unlab'][0]
        pred_class = logits_unlab.max(dim=-1)[1] + self.seen_classes
        gt_classes[unk_mask] = pred_class
        fg_gt_classes = gt_classes[fg_mask]

        return 0.1*self.mlp_model(fg_features,sa_concat_features,fg_gt_classes)

    def get_KL_loss(self,logits_unlab,unk_features):
        for item in self.uno_means:
            if item is not None:
                print(item.shape)
        
        new_uno_means = self.uno_means.copy()
        for index,item in enumerate(new_uno_means):
            if item is None:
                new_uno_means[index] = torch.zeros(2048)
        uno_means = torch.stack(new_uno_means).cuda()
        # print(logits_unlab.shape,unk_features.shape,uno_means.shape)
        center_soft_prob = get_cluster_prob(unk_features,uno_means).softmax(dim=-1)
        logits_soft_unlab = logits_unlab.softmax(dim=-1)
        
        return self.cluster_loss((center_soft_prob+1e-08).log(), logits_soft_unlab)/center_soft_prob.shape[0]
        
        

    def get_new_clustering_loss(self, proposals,input_features):
        
        storage = get_event_storage()
        
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        unk_mask = gt_classes == self.num_classes-1
        unk_features = input_features[unk_mask]
        
        logits_unlab = self.uno_model(unk_features)['logits_unlab'][0]
        if storage.iter <= self.clustering_start_iter:
            pred_class = logits_unlab.max(dim=-1)[1]
            self.uno_feature_store.add(unk_features, pred_class)
        else:
            unk_num = unk_features.shape[0]
            uno_means = torch.stack(self.uno_means).cuda()
            uno_center_expand = uno_means[None,...].expand(unk_num,-1,-1)
            unk_features_expand = unk_features[:,None,:].expand(-1,10,-1)
            unk_sim_scores = self.cos(uno_center_expand,unk_features_expand)
            _ , pred_class = torch.max(unk_sim_scores,dim=-1)
            self.uno_feature_store.add(unk_features,pred_class)
            
        
        c_loss = 0
        
        if storage.iter == self.clustering_start_iter:
            items = self.uno_feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.uno_means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.uno_means[index] = mu
            c_loss = self.get_KL_loss(logits_unlab,unk_features)
            # c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif storage.iter > self.clustering_start_iter:
            if storage.iter % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.uno_feature_store.retrieve(-1)
                new_means = [None for _ in range(10)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.uno_means):
                    if (mean) is not None and new_means[i] is not None:
                        self.uno_means[i] = self.clustering_momentum * mean + \
                                        (1 - self.clustering_momentum) * new_means[i]
            c_loss = self.get_KL_loss(logits_unlab,unk_features)
            # c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return 10 * c_loss

    
    def get_unknown_lang_loss(self, input_features, proposals, input_scores):
        gt_classes = torch.cat([item.gt_classes for item in proposals])
        input_features = input_features[gt_classes == self.num_classes-1]
        input_scores = input_scores[gt_classes == self.num_classes-1]
        
        logits_unlab = self.uno_model(input_features)['logits_unlab'][0]
        labels = logits_unlab.max(-1)[1]
        img_sims = input_features.mm(input_features.T)

        label_embeds = self.clip_process.get_super_features(device=input_features.device)

        latter_10_lang_embeds = label_embeds[labels]
        latter_10_lang_embeds = latter_10_lang_embeds.unsqueeze(1)
        bsame_labels = (labels.T == labels.view(-1, 1)).T

        language_sims = self.compute_language_sims(latter_10_lang_embeds)
        language_sims = language_sims.mean(dim=-1)
        language_sims += self.language_shift
        maskval = 1 + self.language_shift
        return kl_div(language_sims.detach().to(img_sims.device),
                      img_sims,
                      mask=bsame_labels,
                      maskval=maskval,
                      T=1)

    def compute_language_sims(self, language_embeds):
        language_sims = torch.einsum(
            'abe,cbe->acb',
            torch.nn.functional.normalize(language_embeds, dim=-1),
            torch.nn.functional.normalize(language_embeds, dim=-1))
        language_sims = language_sims.reshape(*language_sims.shape[:2], -1)
        return language_sims
    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals, input_features=None, unknown=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        
        scores, proposal_deltas ,sa_semantic,sa_cls_scores= predictions
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            sa_cls_scores,
            proposals,
            self.invalid_class_range,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        losses['loss_new_clustering'] = self.get_new_clustering_loss(proposals,input_features)

        losses['unknown_lang_loss'] = 0.1 * self.get_unknown_lang_loss(input_features, proposals, scores)
        
        # losses['loss_mlp'] = self.get_mlp_loss(sa_semantic,proposals,input_features)
        # losses["loss_semantic"] = self.get_sa_loss3(sa_semantic,proposals,input_features)

        # if input_features is not None:
        #     # losses["loss_cluster_encoder"] = self.get_ae_loss(input_features)
            # losses["loss_clustering"] = self.get_clustering_loss(input_features, proposals)
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def uno_inference(self, input_features):
        outputs = self.uno_model(input_features)
        preds_scores = torch.cat(
            [
                outputs["logits_lab"].unsqueeze(0).expand(1, -1, -1),
                outputs["logits_unlab"],
            ],
            dim=-1,
        )
        # preds_inc = preds_scores.max(dim=-1)[1]
        return preds_scores

    def inference(self, predictions, proposals, input_features):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
            input_features:
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """

        uno_preds = self.uno_inference(input_features)

        # 暂时只用一个头
        uno_preds = uno_preds[0]
        num_prop_per_image = [len(p) for p in proposals]
        uno_preds = uno_preds.split(num_prop_per_image)

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        return uno_fast_rcnn_inference(
            uno_preds,
            boxes,
            scores,
            image_shapes,
            predictions,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.seen_classes
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas,_,_ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _,_,_ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    # def clstr_loss(self, input_features, proposals):
    #     """
    #     Get the foreground input_features, generate distributions for the class,
    #     get probability of each feature from each distribution;
    #     Compute loss: if belonging to a class -> likelihood should be higher
    #                   else -> lower
    #     :param input_features:
    #     :param proposals:
    #     :return:
    #     """
    #     loss = 0
    #     gt_classes = torch.cat([p.gt_classes for p in proposals])
    #     mask = gt_classes != self.num_classes
    #     fg_features = input_features[mask]
    #     classes = gt_classes[mask]
    #     # fg_features = self.ae_model.encoder(fg_features)
    #
    #     # Distribution per class
    #     log_prob = [None for _ in range(self.num_classes + 1)]
    #     # https://github.com/pytorch/pytorch/issues/23780
    #     for cls_index, mu in enumerate(self.means):
    #         if mu is not None:
    #             dist = Normal(loc=mu.cuda(), scale=torch.ones_like(mu.cuda()))
    #             log_prob[cls_index] = dist.log_prob(fg_features).mean(dim=1)
    #             # log_prob[cls_index] = torch.distributions.multivariate_normal. \
    #             #     MultivariateNormal(mu.cuda(), torch.eye(len(mu)).cuda()).log_prob(fg_features)
    #                 # MultivariateNormal(mu, torch.eye(len(mu))).log_prob(fg_features.cpu())
    #             #                     MultivariateNormal(mu[:2], torch.eye(len(mu[:2]))).log_prob(fg_features[:,:2].cpu())
    #         else:
    #             log_prob[cls_index] = torch.zeros((len(fg_features))).cuda()
    #
    #     log_prob = torch.stack(log_prob).T # num_of_fg_proposals x num_of_classes
    #     for i, p in enumerate(log_prob):
    #         weight = torch.ones_like(p) * -1
    #         weight[classes[i]] = 1
    #         p = p * weight
    #         loss += p.mean()
    #     return loss

    # def clstr_loss_l2(self, input_features, proposals):
    #     """
    #     Get the foreground input_features, generate distributions for the class,
    #     get probability of each feature from each distribution;
    #     Compute loss: if belonging to a class -> likelihood should be higher
    #                   else -> lower
    #     :param input_features:
    #     :param proposals:
    #     :return:
    #     """
    #     loss = 0
    #     gt_classes = torch.cat([p.gt_classes for p in proposals])
    #     mask = gt_classes != self.num_classes
    #     fg_features = input_features[mask]
    #     classes = gt_classes[mask]
    #     fg_features = self.ae_model.encoder(fg_features)
    #
    #     for index, feature in enumerate(fg_features):
    #         for cls_index, mu in enumerate(self.means):
    #             if mu is not None and feature is not None:
    #                 mu = mu.cuda()
    #                 if  classes[index] ==  cls_index:
    #                     loss -= F.mse_loss(feature, mu)
    #                 else:
    #                     loss += F.mse_loss(feature, mu)
    #
    #     return loss
