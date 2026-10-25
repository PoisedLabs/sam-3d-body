# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import warnings
from typing import Optional

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F


def _quat_mul_norm(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = q[..., 3:4]
    axis = q[..., :3]
    av = torch.linalg.cross(axis, v, dim=-1)
    aav = torch.linalg.cross(axis, av, dim=-1)
    return v + 2.0 * (av * r + aav)


def _skel_multiply(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    t1 = state1[..., :3]
    q1 = F.normalize(state1[..., 3:7], dim=-1, eps=1e-12)
    s1 = state1[..., 7:8]
    t2 = state2[..., :3]
    q2 = F.normalize(state2[..., 3:7], dim=-1, eps=1e-12)
    s2 = state2[..., 7:8]
    t_res = t1 + s1 * _quat_rotate(q1, t2)
    q_res = _quat_mul_norm(q1, q2)
    s_res = s1 * s2
    return torch.cat([t_res, q_res, s_res], dim=-1)

from ..modules import rot6d_to_rotmat
from ..modules.mhr_utils import (
    batchXYZfrom6D,
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    compact_model_params_to_cont_body,
    mhr_param_hand_mask,
)

from ..modules.transformer import FFN

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR

        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")


class MHRHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        mhr_model_path: str = "",
        extra_joint_regressor: str = "",
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
        enable_hand_model=False,
    ):
        super().__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

        # MHR Parameters
        self.model_data_dir = mhr_model_path
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps

        # Buffers to be filled in by model state dict
        self.joint_rotation = nn.Parameter(torch.zeros(127, 3, 3), requires_grad=False)
        self.scale_mean = nn.Parameter(torch.zeros(68), requires_grad=False)
        self.scale_comps = nn.Parameter(torch.zeros(28, 68), requires_grad=False)
        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
        self.hand_pose_mean = nn.Parameter(torch.zeros(54), requires_grad=False)
        self.hand_pose_comps = nn.Parameter(torch.eye(54), requires_grad=False)
        self.hand_joint_idxs_left = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.hand_joint_idxs_right = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.keypoint_mapping = nn.Parameter(
            torch.zeros(308, 18439 + 127), requires_grad=False
        )
        # Some special buffers for the hand-version
        self.right_wrist_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.root_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.local_to_world_wrist = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
        self.nonhand_param_idxs = nn.Parameter(
            torch.zeros(145).long(), requires_grad=False
        )

        # Load MHR itself
        if MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                lod=1,
            )
        else:
            self.mhr = torch.jit.load(
                mhr_model_path,
                map_location=("cuda" if torch.cuda.is_available() else "cpu"),
            )

        for param in self.mhr.parameters():
            param.requires_grad = False

        self._fast_mhr_cache = None
        self._faces_numpy = None
        self._body_keep_mask = None

    def _get_body_keep_mask(self, device: torch.device) -> torch.Tensor:
        if self._body_keep_mask is None or self._body_keep_mask.device != device:
            mask = (~mhr_param_hand_mask).float()
            mask[-3:] = 0.0
            self._body_keep_mask = mask.to(device).unsqueeze(0)
        return self._body_keep_mask

    def _get_faces_numpy(self):
        if self._faces_numpy is None:
            self._faces_numpy = self.faces.detach().cpu().numpy()
        return self._faces_numpy

    def _get_fast_mhr_cache(self, device: torch.device):
        if self._fast_mhr_cache is not None and self._fast_mhr_cache["device"] == device:
            return self._fast_mhr_cache

        self.mhr.to(device)
        mhr_ts = self.mhr
        pmi = mhr_ts.character_torch.skeleton.pmi.to(device)
        sizes = mhr_ts.character_torch.skeleton._pmi_buffer_sizes
        pmi_splits = [(s[0], s[1]) for s in torch.split(pmi, sizes, dim=1)]

        p0 = getattr(mhr_ts.pose_correctives_model.pose_dirs_predictor, "0")
        p2 = getattr(mhr_ts.pose_correctives_model.pose_dirs_predictor, "2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc_w0 = (
                torch.sparse_coo_tensor(
                    p0.sparse_indices.detach().cpu(),
                    p0.sparse_weight.detach().cpu(),
                    list(p0.sparse_shape),
                )
                .to_dense()
                .to(device)
            )
        pc_w2_full = p2.weight.detach().to(device)

        base_shape_full = mhr_ts.character_torch.blend_shape.base_shape.detach().to(device)
        shape_vectors_cpu = mhr_ts.character_torch.blend_shape.shape_vectors.detach().cpu()
        face_vectors_cpu = mhr_ts.face_expressions_model.shape_vectors.detach().cpu()
        shape_vectors_full = shape_vectors_cpu.reshape(45, -1).to(device)
        face_vectors_full = face_vectors_cpu.reshape(72, -1).to(device)

        lbs = mhr_ts.character_torch.linear_blend_skinning
        inv_bind_pose = lbs.inverse_bind_pose.detach().unsqueeze(0).to(device)
        skin_indices_full = lbs.skin_indices_flattened.detach().long().to(device)
        skin_weights_full = lbs.skin_weights_flattened.detach().view(1, -1, 1).to(device)
        vert_indices_full = lbs.vert_indices_flattened.detach().long().to(device)

        kp70_cpu = self.keypoint_mapping[:70].detach().cpu()
        vert_mask = kp70_cpu[:, :18439].abs().sum(dim=0) > 0
        active_verts = torch.nonzero(vert_mask, as_tuple=False).squeeze(1)
        num_active_verts = int(active_verts.numel())

        orig_to_sub = torch.full((18439,), -1, dtype=torch.long)
        orig_to_sub[active_verts] = torch.arange(num_active_verts, dtype=torch.long)

        lbs_vert_cpu = lbs.vert_indices_flattened.detach().cpu().long()
        sub_vert_idx = orig_to_sub[lbs_vert_cpu]
        active_lbs_mask = sub_vert_idx >= 0

        skin_indices_sub = (
            lbs.skin_indices_flattened.detach().cpu().long()[active_lbs_mask].to(device)
        )
        skin_weights_sub = (
            lbs.skin_weights_flattened.detach().cpu()[active_lbs_mask].view(1, -1, 1).to(device)
        )
        vert_indices_sub = sub_vert_idx[active_lbs_mask].to(device)

        base_shape_sub = base_shape_full[active_verts.to(device)]
        shape_vectors_sub = shape_vectors_cpu[:, active_verts, :].reshape(45, -1).to(device)
        face_vectors_sub = face_vectors_cpu[:, active_verts, :].reshape(72, -1).to(device)
        pc_w2_sub = (
            p2.weight.detach()
            .cpu()
            .view(18439, 3, 3000)[active_verts]
            .reshape(num_active_verts * 3, 3000)
            .to(device)
        )
        kp70_map_sub = torch.cat([kp70_cpu[:, active_verts], kp70_cpu[:, 18439:]], dim=1).to(
            device
        )

        self._fast_mhr_cache = {
            "device": device,
            "pmi_splits": pmi_splits,
            "pc_w0": pc_w0,
            "pc_w2_full": pc_w2_full,
            "pc_w2_sub": pc_w2_sub,
            "base_shape_full": base_shape_full,
            "base_shape_sub": base_shape_sub,
            "shape_vectors_full": shape_vectors_full,
            "shape_vectors_sub": shape_vectors_sub,
            "face_vectors_full": face_vectors_full,
            "face_vectors_sub": face_vectors_sub,
            "inv_bind_pose": inv_bind_pose,
            "skin_indices_full": skin_indices_full,
            "skin_weights_full": skin_weights_full,
            "vert_indices_full": vert_indices_full,
            "skin_indices_sub": skin_indices_sub,
            "skin_weights_sub": skin_weights_sub,
            "vert_indices_sub": vert_indices_sub,
            "num_active_verts": num_active_verts,
            "kp70_map_sub": kp70_map_sub,
        }
        return self._fast_mhr_cache

    def _mhr_forward_fast(
        self,
        shape_params: torch.Tensor,
        model_params: torch.Tensor,
        expr_params: Optional[torch.Tensor] = None,
        return_vertices: bool = True,
    ):
        cache = self._get_fast_mhr_cache(shape_params.device)
        B = shape_params.shape[0]

        padded = torch.cat([model_params, torch.zeros_like(shape_params)], dim=1)
        joint_params = self.mhr.character_torch.model_parameters_to_joint_parameters(padded)
        local_ss = self.mhr.character_torch.joint_parameters_to_local_skeleton_state(
            joint_params
        )

        skel_state = local_ss.clone()
        for source, target in cache["pmi_splits"]:
            s1 = skel_state.index_select(-2, target)
            s2 = skel_state.index_select(-2, source)
            skel_state.index_copy_(-2, source, _skel_multiply(s1, s2))

        pose_6d = self.mhr.pose_correctives_model._pose_features_from_joint_params(
            joint_params
        )
        pc_hidden = F.relu(F.linear(pose_6d, cache["pc_w0"]))

        joint_state = _skel_multiply(skel_state, cache["inv_bind_pose"])
        q_joint = F.normalize(joint_state[..., 3:7], dim=-1, eps=1e-12)

        if not return_vertices:
            V_sub = cache["num_active_verts"]
            unposed = cache["base_shape_sub"].unsqueeze(0) + (
                shape_params @ cache["shape_vectors_sub"]
            ).view(B, V_sub, 3)
            if expr_params is not None:
                unposed = unposed + (expr_params @ cache["face_vectors_sub"]).view(
                    B, V_sub, 3
                )
            unposed = unposed + F.linear(pc_hidden, cache["pc_w2_sub"]).view(B, V_sub, 3)

            js = joint_state.index_select(-2, cache["skin_indices_sub"])
            qj = q_joint.index_select(-2, cache["skin_indices_sub"])
            vp = unposed.index_select(-2, cache["vert_indices_sub"])
            transformed = (
                js[..., :3] + _quat_rotate(qj, js[..., 7:8] * vp)
            ) * cache["skin_weights_sub"]
            skinned_sub = torch.zeros_like(unposed).index_add(
                -2, cache["vert_indices_sub"], transformed
            )
            return None, skinned_sub, skel_state, cache
        else:
            unposed = cache["base_shape_full"].unsqueeze(0) + (
                shape_params @ cache["shape_vectors_full"]
            ).view(B, 18439, 3)
            if expr_params is not None:
                unposed = unposed + (expr_params @ cache["face_vectors_full"]).view(
                    B, 18439, 3
                )
            unposed = unposed + F.linear(pc_hidden, cache["pc_w2_full"]).view(B, 18439, 3)

            js = joint_state.index_select(-2, cache["skin_indices_full"])
            qj = q_joint.index_select(-2, cache["skin_indices_full"])
            vp = unposed.index_select(-2, cache["vert_indices_full"])
            transformed = (
                js[..., :3] + _quat_rotate(qj, js[..., 7:8] * vp)
            ) * cache["skin_weights_full"]
            skinned_full = torch.zeros_like(unposed).index_add(
                -2, cache["vert_indices_full"], transformed
            )
            return skinned_full, None, skel_state, cache

    def get_zero_pose_init(self, factor=1.0):
        # Initialize pose token with zero-initialized learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        weights = torch.zeros(1, self.npose)
        weights[:, : 6 + self.body_cont_dim] = torch.cat(
            [
                torch.FloatTensor([1, 0, 0, 0, 1, 0]),
                compact_model_params_to_cont_body(torch.zeros(1, 133)).squeeze()
                * factor,
            ],
            dim=0,
        )
        return weights

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        both_hands_cont = (
            self.hand_pose_mean
            + hand_pose_params.unflatten(1, (2, self.num_hand_pose_comps))
            @ self.hand_pose_comps
        )
        both_hands_model_params = compact_cont_to_model_params_hand(both_hands_cont)

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left] = both_hands_model_params[:, 0]
        full_pose_params[:, self.hand_joint_idxs_right] = both_hands_model_params[:, 1]

        return full_pose_params  # B x 136

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        do_pcblend=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
        scale_offsets=None,
        vertex_offsets=None,
        slim_keypoints=False,
    ):
        dev_type = (
            shape_params.device.type
            if shape_params.device.type in ("cuda", "mps")
            else "cpu"
        )
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            global_trans = global_trans.float()
            global_rot = global_rot.float()
            body_pose_params = body_pose_params.float()
            if hand_pose_params is not None:
                hand_pose_params = hand_pose_params.float()
            scale_params = scale_params.float()
            shape_params = shape_params.float()
            if expr_params is not None:
                expr_params = expr_params.float()

            if self.enable_hand_model:
                # Transfer wrist-centric predictions to the body.
                global_rot_ori = global_rot.clone()
                global_trans_ori = global_trans.clone()
                global_rot = roma.rotmat_to_euler(
                    "xyz",
                    roma.euler_to_rotmat("xyz", global_rot_ori)
                    @ self.local_to_world_wrist,
                )
                global_trans = (
                    -(
                        roma.euler_to_rotmat("xyz", global_rot)
                        @ (self.right_wrist_coords - self.root_coords)
                        + self.root_coords
                    )
                    + global_trans_ori
                )

            body_pose_params = body_pose_params[..., :130]

            # Convert from scale and shape params to actual scales and vertices
            ## Add singleton batches in case...
            if len(scale_params.shape) == 1:
                scale_params = scale_params[None]
            if len(shape_params.shape) == 1:
                shape_params = shape_params[None]
            ## Convert scale...
            scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
            if scale_offsets is not None:
                scales = scales + scale_offsets

            # Now, figure out the pose.
            ## 10 here is because it's more stable to optimize global translation in meters.
            full_pose_params = torch.cat(
                [global_trans * 10, global_rot, body_pose_params], dim=1
            )  # B x 127
            ## Put in hands
            if hand_pose_params is not None:
                full_pose_params = self.replace_hands_in_pose(
                    full_pose_params, hand_pose_params
                )
            model_params = torch.cat([full_pose_params, scales], dim=1)

            if self.enable_hand_model:
                # Zero out non-hand parameters
                model_params[:, self.nonhand_param_idxs] = 0

            curr_skinned_verts_sub = None
            fast_cache = None
            if not MOMENTUM_ENABLED and (
                shape_params.device.type == "mps" or slim_keypoints
            ):
                (
                    curr_skinned_verts,
                    curr_skinned_verts_sub,
                    curr_skel_state,
                    fast_cache,
                ) = self._mhr_forward_fast(
                    shape_params,
                    model_params,
                    expr_params,
                    return_vertices=not slim_keypoints,
                )
            else:
                curr_skinned_verts, curr_skel_state = self.mhr(
                    shape_params, model_params, expr_params
                )
            curr_joint_coords, curr_joint_quats, _ = torch.split(
                curr_skel_state, [3, 4, 1], dim=2
            )
            if curr_skinned_verts is not None:
                curr_skinned_verts = curr_skinned_verts / 100
            if curr_skinned_verts_sub is not None:
                curr_skinned_verts_sub = curr_skinned_verts_sub / 100
            curr_joint_coords = curr_joint_coords / 100
            curr_joint_rots = (
                roma.unitquat_to_rotmat(curr_joint_quats)
                if return_joint_rotations
                else None
            )

            # Prepare returns
            to_return = [curr_skinned_verts]
            if return_keypoints:
                if slim_keypoints and curr_skinned_verts_sub is not None:
                    model_vert_joints_sub = torch.cat(
                        [curr_skinned_verts_sub, curr_joint_coords], dim=1
                    )
                    model_keypoints_pred = (
                        fast_cache["kp70_map_sub"] @ model_vert_joints_sub
                    )
                else:
                    # Get sapiens 308 keypoints
                    model_vert_joints = torch.cat(
                        [curr_skinned_verts, curr_joint_coords], dim=1
                    )  # B x (num_verts + 127) x 3
                    model_keypoints_pred = self.keypoint_mapping @ model_vert_joints

                if self.enable_hand_model:
                    # Zero out everything except for the right hand
                    model_keypoints_pred[:, :21] = 0
                    model_keypoints_pred[:, 42:] = 0

                to_return = to_return + [model_keypoints_pred]
            if return_joint_coords:
                to_return = to_return + [curr_joint_coords]
            if return_model_params:
                to_return = to_return + [model_params]
            if return_joint_rotations:
                to_return = to_return + [curr_joint_rots]

            if isinstance(to_return, list) and len(to_return) == 1:
                return to_return[0]
            else:
                return tuple(to_return)

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        do_pcblend=True,
        slim_keypoints=False,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
        """
        batch_size = x.shape[0]
        pred = self.proj(x).float()
        dev_type = x.device.type if x.device.type in ("cuda", "mps") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            if init_estimate is not None:
                pred = pred + init_estimate.float()

            # From pred, we want to pull out individual predictions.

            ## First, get globals
            ### Global rotation is first 6.
            count = 6
            global_rot_6d = pred[:, :count]
            global_rot_euler = batchXYZfrom6D(global_rot_6d).flip(-1)  # B x 3
            global_trans = torch.zeros_like(global_rot_euler)

            ## Next, get body pose.
            ### Hold onto raw, continuous version for iterative correction.
            pred_pose_cont = pred[:, count : count + self.body_cont_dim]
            count += self.body_cont_dim
            ### Convert to eulers (and trans), zeroing hands and jaw without boolean mask syncs
            pred_pose_euler = compact_cont_to_model_params_body(
                pred_pose_cont
            ) * self._get_body_keep_mask(pred_pose_cont.device)

            ## Get remaining parameters
            pred_shape = pred[:, count : count + self.num_shape_comps]
            count += self.num_shape_comps
            pred_scale = pred[:, count : count + self.num_scale_comps]
            count += self.num_scale_comps
            pred_hand = pred[:, count : count + self.num_hand_comps * 2]
            count += self.num_hand_comps * 2
            pred_face = pred[:, count : count + self.num_face_comps] * 0
            count += self.num_face_comps

            # Run everything through mhr
            output = self.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot_euler,
                body_pose_params=pred_pose_euler,
                hand_pose_params=pred_hand,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face,
                do_pcblend=do_pcblend,
                return_keypoints=True,
                return_joint_coords=not slim_keypoints,
                return_model_params=True,
                return_joint_rotations=not slim_keypoints,
                slim_keypoints=slim_keypoints,
            )

            # Some existing code to get joints and fix camera system
            if slim_keypoints:
                verts, j3d, mhr_model_params = output
                jcoords = None
                joint_global_rots = None
            else:
                verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
            j3d = j3d[:, :70]  # 308 --> 70 keypoints

            if verts is not None:
                verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1  # Camera system difference
            if jcoords is not None:
                jcoords[..., [1, 2]] *= -1

            # Prep outputs
            output = {
                "pred_pose_raw": torch.cat(
                    [global_rot_6d, pred_pose_cont], dim=1
                ),  # Both global rot and continuous pose
                "pred_pose_rotmat": None,  # This normally used for mhr pose param rotmat supervision.
                "global_rot": global_rot_euler,
                "body_pose": pred_pose_euler,  # Unused during training
                "shape": pred_shape,
                "scale": pred_scale,
                "hand": pred_hand,
                "face": pred_face,
                "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                "pred_vertices": (
                    verts.reshape(batch_size, -1, 3) if verts is not None else None
                ),
                "pred_joint_coords": (
                    jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
                ),
                "faces": self._get_faces_numpy(),
                "joint_global_rots": joint_global_rots,
                "mhr_model_params": mhr_model_params,
            }

            return output
