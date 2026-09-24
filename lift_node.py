# -*- coding: utf-8 -*-
"""Standalone learned latent lift for MiniMax H3 (no re-sampling).

Derivative work of the comfyui-SelfLift node pack:
  repo:   https://github.com/facok/comfyui-SelfLift  (author: facok)
  paper:  "SelfLift" SelfLift-zero, arXiv:2609.02036
This file only wraps the pack's own h3_upscaler.learned_latent_lift and
selflift.artifact_aware_consistency_lift as a standalone ComfyUI node; all
core logic belongs to the original authors. Same license as the host pack.

Exposes h3_upscaler.learned_latent_lift as a ComfyUI node so a natively
sampled latent can be lifted to a higher target resolution WITHOUT the
SelfLift two-stage sampler: deterministic, composition-preserving, no seed
re-roll. Typical chain: KSampler(native) -> SelfLiftH3LatentLift -> VAEDecode.

Optional SelfLift-zero correction (rho > 0): blend the top-rho fraction of
highest-risk locations toward the pixel-VAE anchor (decode -> bicubic upscale
-> re-encode), replicating artifact_aware_consistency_lift from the sampler's
transition stage. H3 latent scale_factor is 1.0, so the sampled latent is
already in VAE space and no process_in/out round trip is needed.
"""
from . import h3_upscaler
from . import selflift


class SelfLiftH3LatentLift:
    @classmethod
    def INPUT_TYPES(cls):
        models = h3_upscaler.list_upscaler_models()
        if not models:  # keep the combo valid even if the folder is empty
            models = ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]
        return {
            "required": {
                "latent_image": ("LATENT", {
                    "tooltip": "Native-resolution H3 latent to lift (VAE latent space)."}),
                "upscaler_model": (models,),
                "scale": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Spatial lift factor applied to the latent grid "
                               "(pixel size lifts by the same factor)."}),
                "rho": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of highest-risk locations corrected toward "
                               "the pixel-VAE anchor. 0 = pure learned lift "
                               "(deterministic, composition-preserving)."}),
                "w_min": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Correction-strength floor for the selected locations."}),
                "w_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Correction-strength ceiling for the selected locations."}),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "H3 video VAE. Required when rho > 0 (builds the "
                               "pixel-VAE anchor); ignored at rho = 0."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "lift"
    CATEGORY = "SelfLift"

    def lift(self, latent_image, upscaler_model, scale, rho=0.0, w_min=0.5, w_max=1.0, vae=None):
        if not 0.0 <= rho <= 1.0:
            raise ValueError("SelfLiftH3LatentLift: rho must be between 0 and 1")
        if not 0.0 <= w_min <= w_max <= 1.0:
            raise ValueError("SelfLiftH3LatentLift: weights must satisfy 0 <= w_min <= w_max <= 1")
        if rho > 0.0 and vae is None:
            raise ValueError("SelfLiftH3LatentLift: rho > 0 requires the video VAE input "
                             "(pixel-VAE anchor); set rho = 0 or connect the VAE")
        z = latent_image["samples"]
        # H3 sampler output is NestedTensor((video_stream, audio_stream, ...)) —
        # lift only the video stream, pass every other stream through untouched.
        if hasattr(z, "tensors") and hasattr(z, "is_nested"):  # comfy.nested_tensor.NestedTensor
            streams = list(z.tensors)
            streams[0] = self._lift_one(streams[0], upscaler_model, scale, rho, w_min, w_max, vae)
            out = type(z)(streams)
        else:
            out = self._lift_one(z, upscaler_model, scale, rho, w_min, w_max, vae)
        d = {k: v for k, v in latent_image.items() if k != "samples"}
        d["samples"] = out
        return (d,)

    @staticmethod
    def _lift_one(z, upscaler_model, scale, rho, w_min, w_max, vae):
        if z.ndim < 4:
            raise ValueError("SelfLiftH3LatentLift: expected a 4D/5D latent")
        h, w = int(z.shape[-2]), int(z.shape[-1])
        H = max(8, int(round(h * scale)))
        W = max(8, int(round(w * scale)))
        z_lat = h3_upscaler.learned_latent_lift(z, (H, W), upscaler_model)
        if rho <= 0.0 or w_max <= 0.0:
            return z_lat
        # SelfLift-zero correction: top-rho locations pulled toward the pixel anchor.
        z_pix = selflift._pixel_anchor_video(z, vae, (H, W))
        return selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max)


NODE_CLASS_MAPPINGS = {"SelfLiftH3LatentLift": SelfLiftH3LatentLift}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftH3LatentLift": "SelfLift H3 Latent Lift (standalone)",
}
