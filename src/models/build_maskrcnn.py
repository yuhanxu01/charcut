import torchvision
import torch.nn as nn

def build_maskrcnn(num_classes=2, pretrained=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_feats_mask, hidden, num_classes)
    return model
