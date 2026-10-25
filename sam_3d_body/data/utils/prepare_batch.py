# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from sam_3d_body.data.transforms.bbox_utils import (
    bbox_xyxy2cs,
    fix_aspect_ratio,
    get_udp_warp_matrix,
    get_warp_matrix,
)


class NoCollate:
    def __init__(self, data):
        self.data = data


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]
    has_mask = masks is not None

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if has_mask:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    else:
        in_h, in_w = batch["img"].shape[-2:]
        batch["mask"] = torch.zeros(
            (1, max_num_person, 1, in_h, in_w),
            dtype=batch["img"].dtype,
            device=batch["img"].device,
        )
    batch["has_mask"] = has_mask
    batch["is_normalized"] = True
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch


def prepare_image_batch(
    images: Sequence[np.ndarray],
    transform=None,
    boxes_list: Optional[Sequence[np.ndarray]] = None,
    masks_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    masks_score_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    cam_int: Optional[torch.Tensor] = None,
    input_size: Tuple[int, int] = (512, 512),
    padding: float = 1.25,
    aspect_ratio: float = 0.75,
    use_udp: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    """Prepare a multi-frame batch ``[B, N_person, 3, H, W]`` for batched SAM 3D Body inference."""
    batch_size = len(images)
    if batch_size == 0:
        raise ValueError("images must be non-empty")

    has_mask = masks_list is not None and any(m is not None for m in masks_list)
    resolved_boxes = []
    counts = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if boxes_list is not None and boxes_list[i] is not None and len(boxes_list[i]) > 0:
            b = np.asarray(boxes_list[i], dtype=np.float32).reshape(-1, 4)
        else:
            b = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
        resolved_boxes.append(b)
        counts.append(b.shape[0])

    # Fast path when no masks are used and every frame has the same number of persons (e.g. 1)
    if not has_mask and len(set(counts)) == 1:
        num_person = counts[0]
        dst_w, dst_h = int(input_size[0]), int(input_size[1])
        warp_size = (dst_w, dst_h)

        warped_uint8 = np.empty(
            (batch_size, num_person, dst_h, dst_w, 3), dtype=np.uint8
        )
        bbox_arr = np.empty((batch_size, num_person, 4), dtype=np.float32)
        bbox_center_arr = np.empty((batch_size, num_person, 2), dtype=np.float32)
        bbox_scale_arr = np.empty((batch_size, num_person, 2), dtype=np.float32)
        ori_img_size_arr = np.empty((batch_size, num_person, 2), dtype=np.float32)
        img_size_arr = np.broadcast_to(
            np.array([dst_w, dst_h], dtype=np.float32), (batch_size, num_person, 2)
        ).copy()
        affine_trans_arr = np.empty((batch_size, num_person, 2, 3), dtype=np.float32)
        cam_int_arr = np.empty((batch_size, 3, 3), dtype=np.float32)

        last_key = None
        last_warp = None
        last_center = None
        last_scale = None

        for b_idx, img in enumerate(images):
            h, w = img.shape[:2]
            focal = float((h * h + w * w) ** 0.5)
            cam_int_arr[b_idx] = (
                (focal, 0.0, w / 2.0),
                (0.0, focal, h / 2.0),
                (0.0, 0.0, 1.0),
            )
            boxes = resolved_boxes[b_idx]
            for p_idx in range(num_person):
                box = boxes[p_idx]
                bbox_arr[b_idx, p_idx] = box
                ori_img_size_arr[b_idx, p_idx] = (float(w), float(h))
                key = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    dst_w,
                    dst_h,
                )
                if key == last_key:
                    center, scale, warp_mat = last_center, last_scale, last_warp
                else:
                    center, scale = bbox_xyxy2cs(box, padding=padding)
                    scale = fix_aspect_ratio(scale, aspect_ratio=aspect_ratio)
                    scale = fix_aspect_ratio(scale, aspect_ratio=dst_w / dst_h)
                    if use_udp:
                        warp_mat = get_udp_warp_matrix(
                            center, scale, 0.0, output_size=warp_size
                        )
                    else:
                        warp_mat = get_warp_matrix(
                            center, scale, 0.0, output_size=warp_size
                        )
                    last_key = key
                    last_center, last_scale, last_warp = center, scale, warp_mat

                bbox_center_arr[b_idx, p_idx] = center
                bbox_scale_arr[b_idx, p_idx] = scale
                affine_trans_arr[b_idx, p_idx] = warp_mat
                cv2.warpAffine(
                    img,
                    warp_mat,
                    warp_size,
                    dst=warped_uint8[b_idx, p_idx],
                    flags=cv2.INTER_LINEAR,
                )

        target_dev = torch.device(device) if device is not None else torch.device("cpu")
        img_tensor = (
            torch.from_numpy(warped_uint8)
            .to(device=target_dev)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )

        batch = {
            "img": img_tensor,
            "img_size": torch.from_numpy(img_size_arr).to(device=target_dev),
            "ori_img_size": torch.from_numpy(ori_img_size_arr).to(device=target_dev),
            "bbox_center": torch.from_numpy(bbox_center_arr).to(device=target_dev),
            "bbox_scale": torch.from_numpy(bbox_scale_arr).to(device=target_dev),
            "bbox": torch.from_numpy(bbox_arr).to(device=target_dev),
            "affine_trans": torch.from_numpy(affine_trans_arr).to(device=target_dev),
            "mask": torch.zeros(
                (batch_size, num_person, 1, dst_h, dst_w),
                dtype=torch.float32,
                device=target_dev,
            ),
            "mask_score": torch.zeros(
                (batch_size, num_person), dtype=torch.float32, device=target_dev
            ),
            "has_mask": False,
            "is_normalized": True,
            "person_valid": torch.ones(
                (batch_size, num_person), dtype=torch.float32, device=target_dev
            ),
            "cam_int": (
                cam_int.to(device=target_dev, dtype=torch.float32).expand(
                    batch_size, -1, -1
                )
                if cam_int is not None
                else torch.from_numpy(cam_int_arr).to(device=target_dev)
            ),
            "img_ori": [NoCollate(im) for im in images],
        }
        return batch

    # Fallback path: collate per-image prepare_batch dicts
    per_image_batches = []
    for i, img in enumerate(images):
        per_image_batches.append(
            prepare_batch(
                img,
                transform,
                resolved_boxes[i],
                masks=masks_list[i] if masks_list is not None else None,
                masks_score=(
                    masks_score_list[i] if masks_score_list is not None else None
                ),
                cam_int=cam_int[i : i + 1] if (cam_int is not None and cam_int.shape[0] == batch_size) else cam_int,
            )
        )

    batch = {}
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
        "person_valid",
        "cam_int",
    ]:
        if key in per_image_batches[0]:
            batch[key] = torch.cat([b[key] for b in per_image_batches], dim=0)
    batch["has_mask"] = has_mask
    batch["is_normalized"] = True
    batch["img_ori"] = [b["img_ori"][0] for b in per_image_batches]
    if device is not None:
        target_dev = torch.device(device)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device=target_dev)
    return batch
