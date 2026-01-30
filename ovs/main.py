import logging
import os.path
import time
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.load import load_dataset
from ovs.pipeline import Pipeline
from util.args import parse_args
from util.miou import ShowSegmentResult, cam_to_label, get_true_label

import torch
import numpy as np
import random

from ovs.label_utils import refine_label_convex_optimization


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():

    cfg = parse_args()
    print(f"Config for inference:\n{cfg}")
    cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]
    print("Initializing model...")
    pipe = Pipeline(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    pipe.eval()
    print("Model loaded and set to evaluation mode.")
    dataset, bg_text_features, fg_text_features = load_dataset(cfg, pipe.clip)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fg_text_features, bg_text_features = fg_text_features.to(device), bg_text_features.to(device)
    show_segment_result_evaluation = ShowSegmentResult(num_classes=dataset.category_number, ignore_labels=[cfg.ignore_labels])
    consumption_time = 0

    with torch.no_grad():
        progress_bar_iterator = tqdm(enumerate(dataloader),
                                     total=len(dataloader),
                                     desc="Inference Progress")
        for i, (ori_img, img, combined_path_str) in progress_bar_iterator:
            combined_path_str = combined_path_str[0]
            folder_name, base_name = os.path.split(combined_path_str)
            ori_height, ori_width = ori_img.shape[-2], ori_img.shape[-1]
            ori_img, img = ori_img.to(device), img.to(device)
            torch.cuda.synchronize()
            start_time = time.time()
            pred_mask_raw, final_score, pred_mask_raw2, final_score2 = pipe(ori_img, img, fg_text_features, bg_text_features)
            pred_mask_processed = F.interpolate(pred_mask_raw[None], size=(ori_height, ori_width), mode='bilinear', align_corners=False)[0]
            pred_mask_processed = pred_mask_processed / (pred_mask_processed.amax((-1, -2), keepdim=True) + 1e-5) * \
                                  final_score[..., None, None]
            weight_mask, arg_mask, all_values = cam_to_label(pred_mask_processed[None].clone(), cls_label=final_score[None],
                                       bkg_thre=cfg.bkg_thre, cls_thre=cfg.score_threshold,
                                       is_normalize=cfg.is_normalize)
            pred_mask_processed2 = \
            F.interpolate(pred_mask_raw2[None], size=(ori_height, ori_width), mode='bilinear', align_corners=False)[0]
            pred_mask_processed2 = pred_mask_processed2 / (pred_mask_processed2.amax((-1, -2), keepdim=True) + 1e-5) * \
                                  final_score2[..., None, None]
            weight_mask2, arg_mask2, all_values2 = cam_to_label(pred_mask_processed2[None].clone(), cls_label=final_score2[None],
                                       bkg_thre=cfg.bkg_thre, cls_thre=cfg.score_threshold,
                                       is_normalize=cfg.is_normalize)
            corrected_label_indexed, pseudo_label_for_loss, fb_region_mask = refine_label_convex_optimization(
                prob_map_1=all_values,
                prob_map_2=all_values2,
                combined_path_str=combined_path_str,
                dataset=dataset,
                cfg=cfg
            )

            torch.cuda.synchronize()
            end_time = time.time()
            consumption_time += end_time - start_time

            if hasattr(cfg, "save_path"):
                color_output_dir = os.path.join(cfg.save_path, "inference_results_color", folder_name)
                os.makedirs(color_output_dir, exist_ok=True)
                color_save_path = os.path.join(color_output_dir, f"{base_name}.png")
                result_numpy_gray = corrected_label_indexed.cpu().numpy().astype(np.uint8)
                colored_result_bgr = dataset.palette[result_numpy_gray]
                cv2.imwrite(color_save_path, colored_result_bgr)
                progress_bar_iterator.set_postfix(saved_to=f"{folder_name}/{base_name}.png")
            label_true = get_true_label(combined_path_str, cfg.label_root, target_shape=pseudo_label_for_loss.shape)
            show_segment_result_evaluation.add_prediction(label_true.astype(np.int64), pseudo_label_for_loss.cpu().numpy())
    result = show_segment_result_evaluation.calculate()
    output = (
            f"pAcc: {result['pAcc']:.4f}, \n"
            f"mAcc: {result['mAcc']:.4f}, \n"
            f"mIoU: {result['mIoU']:.4f},  \n"
            "IoU:\n" + "\n".join([f"{i:>2}: {iou:.4f}" for i, iou in result['IoU'].items()]) + "\n"
            f"consumption time: {consumption_time:.4f} s\n"
    )
    print("".join(output))

if __name__ == '__main__':
    main()