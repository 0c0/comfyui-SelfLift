# -*- coding: utf-8 -*-
"""File persistence for MiniMax H3 NestedTensor latents (video + audio streams).

Derivative work of the comfyui-SelfLift node pack:
  repo:   https://github.com/facok/comfyui-SelfLift  (author: facok)
  paper:  "SelfLift" SelfLift-zero, arXiv:2609.02036
Serialization glue only — no algorithmic content of the original pack is
reimplemented here. Same license as the host pack.

ComfyUI's built-in SaveLatent calls samples["samples"].contiguous(), which the
comfy.nested_tensor.NestedTensor wrapper does not implement, so H3 two-stream
latents cannot go through it. These nodes serialize each stream as its own
safetensors key (stream_0 / stream_1 / ...) plus a format marker, and rebuild
the NestedTensor on load. Purpose: cache the expensive native-resolution
sample once, then iterate lift variants (rho / scale / upscaler) against the
frozen latent without re-sampling.
"""
import json
import os

import torch

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy.utils import save_torch_file

FORMAT_MARKER = torch.IntTensor([0x5333])  # "S3" marker distinguishing our files


class SelfLiftH3LatentSave:
    """Save an H3 NestedTensor latent (or any plain LATENT) as .latent.safetensors."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename": ("STRING", {
                    "default": "latent/H3LAT_cache",
                    "tooltip": "Path relative to the output directory; written as "
                               "<filename>.latent.safetensors (deterministic overwrite, "
                               "no counter — stable cache key for batch runners)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "SelfLift"

    def save(self, latent, filename, prompt=None, extra_pnginfo=None):
        z = latent["samples"]
        if hasattr(z, "tensors") and hasattr(z, "is_nested"):
            streams = [t.detach().cpu().contiguous() for t in z.tensors]
        else:
            streams = [z.detach().cpu().contiguous()]
        out = {("stream_%d" % i): t for i, t in enumerate(streams)}
        out["format"] = FORMAT_MARKER
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt if isinstance(prompt, str) else json.dumps(prompt)
        if extra_pnginfo:
            for k, v in extra_pnginfo.items():
                metadata[k] = v if isinstance(v, str) else json.dumps(v)
        full_output_folder, base, _counter, _sub, _pfx = folder_paths.get_save_image_path(
            filename, self.output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        path = os.path.join(full_output_folder, base + ".latent.safetensors")
        save_torch_file(out, path, metadata=metadata or None)
        return {"ui": {"latents": [{"filename": os.path.basename(path),
                                    "subfolder": os.path.dirname(filename) or "",
                                    "type": "output"}]},
                "result": (latent,)}


class SelfLiftH3LatentLoad:
    """Load a file written by SelfLiftH3LatentSave back into a LATENT (NestedTensor)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": r"E:/ai/ComfyUI-aki-v3/ComfyUI/output/latent/H3LAT_cache.latent.safetensors",
                    "tooltip": "Absolute path to the .latent.safetensors cache file."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"
    CATEGORY = "SelfLift"

    @classmethod
    def IS_CHANGED(cls, path):
        """Cache key = file content hash, not graph structure.

        Without this, ComfyUI serves a cached output when the same graph is
        resubmitted against a *different* file at the same path (observed:
        stale latent silently reused after the cache file was regenerated).
        """
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self, path):
        from safetensors.torch import load_file
        if not os.path.isfile(path):
            raise FileNotFoundError("SelfLiftH3LatentLoad: no such file: %s" % path)
        d = load_file(path, device="cpu")
        if "format" not in d:
            raise ValueError("SelfLiftH3LatentLoad: %s was not written by "
                             "SelfLiftH3LatentSave" % path)
        i = 0
        streams = []
        while "stream_%d" % i in d:
            streams.append(d["stream_%d" % i].float())
            i += 1
        if not streams:
            raise ValueError("SelfLiftH3LatentLoad: no stream_* tensors in %s" % path)
        samples = NestedTensor(streams) if len(streams) > 1 else streams[0]
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "SelfLiftH3LatentSave": SelfLiftH3LatentSave,
    "SelfLiftH3LatentLoad": SelfLiftH3LatentLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftH3LatentSave": "SelfLift H3 Latent Save",
    "SelfLiftH3LatentLoad": "SelfLift H3 Latent Load",
}
