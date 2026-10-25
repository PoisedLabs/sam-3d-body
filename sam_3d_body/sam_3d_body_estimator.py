# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import List, Optional, Sequence, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch, prepare_image_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, self.device)
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    @torch.no_grad()
    def process_image_batch(
        self,
        images: Sequence[Union[str, np.ndarray]],
        bboxes: Optional[Sequence[Optional[np.ndarray]]] = None,
        masks: Optional[Sequence[Optional[np.ndarray]]] = None,
        cam_int: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: Optional[str] = None,
    ) -> List[List[dict]]:
        """
        Perform batched multi-frame model prediction across a sequence of images.

        Args:
            images: Sequence of input images (paths or RGB numpy arrays).
            bboxes: Optional sequence of per-image bounding boxes ``(N_i, 4)``.
            masks: Optional sequence of per-image masks.
            cam_int: Optional camera intrinsics tensor ``(1, 3, 3)`` or ``(B, 3, 3)``.
            batch_size: Optional mini-batch size for chunking large sequences.
            det_cat_id: Detection category ID when ``self.detector`` is used.
            bbox_thr: Bounding box threshold when ``self.detector`` is used.
            nms_thr: NMS threshold when ``self.detector`` is used.
            use_mask: Whether to compute masks with ``self.sam`` when available.
            inference_type: ``"body"`` (default for batched inference), ``"full"``, or ``"hand"``.

        Returns:
            List of per-image prediction lists (each element matches ``process_one_image`` output).
        """
        if not images:
            return []

        resolved_inference_type = (
            inference_type or os.environ.get("SAM3D_INFERENCE_TYPE", "body")
        )

        if batch_size is not None and batch_size > 0 and len(images) > batch_size:
            results: List[List[dict]] = []
            for start in range(0, len(images), batch_size):
                end = start + batch_size
                sub_bboxes = bboxes[start:end] if bboxes is not None else None
                sub_masks = masks[start:end] if masks is not None else None
                sub_cam = (
                    cam_int[start:end]
                    if (cam_int is not None and cam_int.shape[0] == len(images))
                    else cam_int
                )
                results.extend(
                    self.process_image_batch(
                        images[start:end],
                        bboxes=sub_bboxes,
                        masks=sub_masks,
                        cam_int=sub_cam,
                        batch_size=None,
                        det_cat_id=det_cat_id,
                        bbox_thr=bbox_thr,
                        nms_thr=nms_thr,
                        use_mask=use_mask,
                        inference_type=resolved_inference_type,
                    )
                )
            return results

        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []

        rgb_images: List[np.ndarray] = []
        boxes_list: List[np.ndarray] = []
        masks_list: List[Optional[np.ndarray]] = []
        masks_score_list: List[Optional[np.ndarray]] = []
        empty_frame_indices = set()

        for i, img in enumerate(images):
            if isinstance(img, str):
                loaded = load_image(img, backend="cv2", image_format="bgr")
                image_format = "bgr"
            else:
                loaded = img
                image_format = "rgb"
            height, width = loaded.shape[:2]

            frame_bboxes = bboxes[i] if bboxes is not None else None
            if frame_bboxes is not None:
                boxes = np.asarray(frame_bboxes, dtype=np.float32).reshape(-1, 4)
                self.is_crop = True
            elif self.detector is not None:
                bgr_img = (
                    cv2.cvtColor(loaded, cv2.COLOR_RGB2BGR)
                    if image_format == "rgb"
                    else loaded
                )
                boxes = self.detector.run_human_detection(
                    bgr_img,
                    det_cat_id=det_cat_id,
                    bbox_thr=bbox_thr,
                    nms_thr=nms_thr,
                    default_to_full_image=False,
                )
                self.is_crop = True
            else:
                boxes = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
                self.is_crop = False

            if len(boxes) == 0:
                empty_frame_indices.add(i)
                continue

            rgb_img = (
                cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
                if image_format == "bgr"
                else loaded
            )
            rgb_images.append(rgb_img)
            boxes_list.append(boxes)

            frame_masks = masks[i] if masks is not None else None
            if frame_masks is not None:
                m = np.asarray(frame_masks).reshape(-1, height, width, 1).astype(np.uint8)
                ms = np.ones(len(m), dtype=np.float32)
                masks_list.append(m)
                masks_score_list.append(ms)
            elif use_mask and self.sam is not None:
                m, ms = self.sam.run_sam(rgb_img, boxes)
                masks_list.append(m)
                masks_score_list.append(ms)
            else:
                masks_list.append(None)
                masks_score_list.append(None)

        if not rgb_images:
            return [[] for _ in images]

        # If different frames have different numbers of detected people (>1), fall back to per-image
        person_counts = [b.shape[0] for b in boxes_list]
        if len(set(person_counts)) > 1:
            results = []
            valid_cursor = 0
            for i in range(len(images)):
                if i in empty_frame_indices:
                    results.append([])
                else:
                    results.append(
                        self.process_one_image(
                            rgb_images[valid_cursor],
                            bboxes=boxes_list[valid_cursor],
                            masks=masks_list[valid_cursor],
                            cam_int=cam_int,
                            det_cat_id=det_cat_id,
                            bbox_thr=bbox_thr,
                            nms_thr=nms_thr,
                            use_mask=use_mask,
                            inference_type=resolved_inference_type,
                        )
                    )
                    valid_cursor += 1
            return results

        batch = prepare_image_batch(
            rgb_images,
            transform=self.transform,
            boxes_list=boxes_list,
            masks_list=masks_list if any(m is not None for m in masks_list) else None,
            masks_score_list=(
                masks_score_list if any(m is not None for m in masks_list) else None
            ),
            cam_int=cam_int,
            input_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
            device=self.device,
        )

        self.model._initialize_batch(batch)

        if cam_int is not None:
            batch["cam_int"] = cam_int.to(batch["img"]).expand(len(rgb_images), -1, -1).clone()
        elif self.fov_estimator is not None:
            cam_ints = [
                self.fov_estimator.get_cam_intrinsics(im).to(batch["img"])
                for im in rgb_images
            ]
            batch["cam_int"] = torch.cat(cam_ints, dim=0)

        batch_lhand, batch_rhand = None, None
        if resolved_inference_type == "body":
            pose_output = self.model.forward_step(batch, decoder_type="body")
        elif resolved_inference_type == "hand":
            pose_output = self.model.forward_step(batch, decoder_type="hand")
        elif resolved_inference_type == "full":
            outputs = self.model.run_inference(
                rgb_images[0],
                batch,
                inference_type="full",
                transform_hand=self.transform_hand,
                thresh_wrist_angle=self.thresh_wrist_angle,
            )
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            raise ValueError(f"Invalid inference type: {resolved_inference_type}")

        raw_out = (
            pose_output["mhr"]
            if resolved_inference_type != "hand"
            else pose_output["mhr_hand"]
        )
        out_keys = (
            "focal_length",
            "pred_keypoints_3d",
            "pred_keypoints_2d",
            "pred_vertices",
            "pred_cam_t",
            "pred_pose_raw",
            "global_rot",
            "body_pose",
            "hand",
            "scale",
            "shape",
            "face",
            "pred_joint_coords",
            "joint_global_rots",
            "mhr_model_params",
        )
        out = {
            k: (
                raw_out[k].detach().float().cpu().numpy()
                if raw_out[k].dtype == torch.bfloat16
                else raw_out[k].detach().cpu().numpy()
            )
            for k in out_keys
        }
        bboxes_np = batch["bbox"].detach().cpu().numpy()

        if resolved_inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
            lhand_center = batch_lhand["bbox_center"].flatten(0, 1).detach().cpu().numpy()
            lhand_scale = batch_lhand["bbox_scale"].flatten(0, 1).detach().cpu().numpy()
            rhand_center = batch_rhand["bbox_center"].flatten(0, 1).detach().cpu().numpy()
            rhand_scale = batch_rhand["bbox_scale"].flatten(0, 1).detach().cpu().numpy()

        num_valid_frames, num_person = batch["img"].shape[:2]
        del batch, pose_output, raw_out
        valid_results: List[List[dict]] = []
        flat_idx = 0
        for b_idx in range(num_valid_frames):
            frame_out: List[dict] = []
            frame_mask = masks_list[b_idx]
            for p_idx in range(num_person):
                person_dict = {
                    "bbox": bboxes_np[b_idx, p_idx],
                    "focal_length": out["focal_length"][flat_idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][flat_idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][flat_idx],
                    "pred_vertices": out["pred_vertices"][flat_idx],
                    "pred_cam_t": out["pred_cam_t"][flat_idx],
                    "pred_pose_raw": out["pred_pose_raw"][flat_idx],
                    "global_rot": out["global_rot"][flat_idx],
                    "body_pose_params": out["body_pose"][flat_idx],
                    "hand_pose_params": out["hand"][flat_idx],
                    "scale_params": out["scale"][flat_idx],
                    "shape_params": out["shape"][flat_idx],
                    "expr_params": out["face"][flat_idx],
                    "mask": frame_mask[p_idx] if frame_mask is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][flat_idx],
                    "pred_global_rots": out["joint_global_rots"][flat_idx],
                    "mhr_model_params": out["mhr_model_params"][flat_idx],
                }
                if resolved_inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
                    person_dict["lhand_bbox"] = np.array(
                        [
                            float(lhand_center[flat_idx, 0] - lhand_scale[flat_idx, 0] / 2.0),
                            float(lhand_center[flat_idx, 1] - lhand_scale[flat_idx, 1] / 2.0),
                            float(lhand_center[flat_idx, 0] + lhand_scale[flat_idx, 0] / 2.0),
                            float(lhand_center[flat_idx, 1] + lhand_scale[flat_idx, 1] / 2.0),
                        ]
                    )
                    person_dict["rhand_bbox"] = np.array(
                        [
                            float(rhand_center[flat_idx, 0] - rhand_scale[flat_idx, 0] / 2.0),
                            float(rhand_center[flat_idx, 1] - rhand_scale[flat_idx, 1] / 2.0),
                            float(rhand_center[flat_idx, 0] + rhand_scale[flat_idx, 0] / 2.0),
                            float(rhand_center[flat_idx, 1] + rhand_scale[flat_idx, 1] / 2.0),
                        ]
                    )
                frame_out.append(person_dict)
                flat_idx += 1
            valid_results.append(frame_out)

        if not empty_frame_indices:
            return valid_results

        final_results: List[List[dict]] = []
        valid_cursor = 0
        for i in range(len(images)):
            if i in empty_frame_indices:
                final_results.append([])
            else:
                final_results.append(valid_results[valid_cursor])
                valid_cursor += 1
        return final_results
