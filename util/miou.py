import numpy as np
import torch
import cv2
import os

class ShowSegmentResult:

    def __init__(self, num_classes=21, ignore_labels=None):

        if ignore_labels is None:
            ignore_labels = [255]
        self.ignore_labels = ignore_labels
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.result = None

    @staticmethod
    def _fast_hist(label_true, label_pred, num_classes):

        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        )
        return hist.reshape(num_classes, num_classes)

    def add_prediction(self, label_true, label_pred):

        ignore_mask = ~(np.isin(label_true, self.ignore_labels))

        # Apply the mask and flatten the arrays
        label_true_flat = label_true[ignore_mask].flatten()
        label_pred_flat = label_pred[ignore_mask].flatten()

        self.hist += self._fast_hist(label_true_flat, label_pred_flat, self.num_classes)

    def calculate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        _acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(_acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        valid = self.hist.sum(axis=1) > 0
        mean_iu = np.nanmean(iu[valid])
        cls_iu = dict(zip(range(self.num_classes), iu))

        self.result = {"pAcc": acc, "mAcc": acc_cls, "mIoU": mean_iu, "IoU": cls_iu}

        return self.result

    def clear_prediction(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.result = None


def  cam_to_label(cam, cls_label, bkg_thre=0.3, cls_thre=0.4, is_normalize=True):

    b, c, h, w = cam.shape

    cls_label_rep = (cls_label > cls_thre).unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    reshape_cam = cam.reshape(b, c, -1)
    if is_normalize:
        reshape_cam -= reshape_cam.amin(dim=-1, keepdim=True)
        reshape_cam /= reshape_cam.amax(dim=-1, keepdim=True) + 1e-6
    if bkg_thre > 0:
        reshape_cam[reshape_cam < bkg_thre] = 0
        reshape_cam = reshape_cam.reshape(b, c, h, w)

        valid_cam = cls_label_rep * reshape_cam
        max_values, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
        _pseudo_label += 1
        _pseudo_label[max_values == 0] = 0
        all_values = valid_cam[0]
    else:
        reshape_cam = reshape_cam.reshape(b, c, h, w)
        valid_cam = cls_label_rep * reshape_cam
        _pseudo_label = valid_cam.argmax(dim=1)
        max_values = valid_cam.amax(dim=1)
        all_values = valid_cam[0]
    return max_values, _pseudo_label, all_values


def get_true_label(combined_path_str, label_root, target_shape):

    label_true_path = os.path.join(label_root, combined_path_str + '.png')
    label_true_path = label_true_path.replace("_img", "_label")
    label_true = cv2.imread(label_true_path, cv2.IMREAD_GRAYSCALE)

    return label_true

