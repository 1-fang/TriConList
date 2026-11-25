import os
import argparse
import types
import sys
from collections import defaultdict
import torch
from safetensors.torch import save_file

# =============================
# DTensor / FSDP related imports (compatible with different PyTorch versions)
# =============================

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

try:
    from torch.distributed._tensor.placement_types import _Partial
except ImportError:
    _Partial = None

# Layout may not exist in different PyTorch versions, make it optional import
try:
    from torch.distributed._tensor.api import Layout
except ImportError:
    Layout = None

# Register FSDP / DTensor related classes to ensure torch.load can deserialize
safe_globals = [DTensor, Replicate, Shard]
if Layout is not None:
    safe_globals.append(Layout)
if _Partial is not None:
    safe_globals.append(_Partial)

torch.serialization.add_safe_globals(safe_globals)

# Ensure torch.distributed.tensor exposes DTensor for older versions
# Ensure torch.distributed.tensor package exists and exposes DTensor
if "torch.distributed.tensor" not in sys.modules:
    tensor_module = types.ModuleType("torch.distributed.tensor")
    tensor_module.__path__ = []
    sys.modules["torch.distributed.tensor"] = tensor_module

_dist_tensor_mod = sys.modules["torch.distributed.tensor"]
if not hasattr(_dist_tensor_mod, "__path__"):
    _dist_tensor_mod.__path__ = []
if not hasattr(_dist_tensor_mod, "DTensor"):
    setattr(_dist_tensor_mod, "DTensor", DTensor)

# Ensure placement_types module exists for serialized references
if "torch.distributed.tensor.placement_types" not in sys.modules:
    placement_module = types.ModuleType("torch.distributed.tensor.placement_types")
    placement_module.Replicate = Replicate
    placement_module.Shard = Shard
    if _Partial is not None:
        placement_module._Partial = _Partial
    sys.modules["torch.distributed.tensor.placement_types"] = placement_module

# Ensure torch.distributed.tensor._dtensor_spec exists (for checkpoints serialized with new PyTorch)
if "torch.distributed.tensor._dtensor_spec" not in sys.modules:
    legacy_spec_module = None
    try:
        from torch.distributed.tensor import _dtensor_spec as _tensor_dtensor_spec

        legacy_spec_module = _tensor_dtensor_spec
    except (ImportError, ModuleNotFoundError):
        try:
            from torch.distributed._tensor import (
                _dtensor_spec as _legacy_dtensor_spec_mod,
            )

            legacy_spec_module = _legacy_dtensor_spec_mod
        except (ImportError, ModuleNotFoundError):
            legacy_spec_module = None

    if legacy_spec_module is not None:
        alias_module = types.ModuleType("torch.distributed.tensor._dtensor_spec")
        for attr in dir(legacy_spec_module):
            if not attr.startswith("__"):
                setattr(alias_module, attr, getattr(legacy_spec_module, attr))
    else:
        alias_module = types.ModuleType("torch.distributed.tensor._dtensor_spec")

        class _DummyDTensorSpec:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _DummyTensorMeta:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _DummyPlacementSpec:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        setattr(alias_module, "DTensorSpec", _DummyDTensorSpec)
        setattr(alias_module, "TensorMeta", _DummyTensorMeta)
        setattr(alias_module, "PlacementSpec", _DummyPlacementSpec)

    sys.modules["torch.distributed.tensor._dtensor_spec"] = alias_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge FSDP shards and save as safetensors"
    )
    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory that contains model_world_size_*_rank_*.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output safetensors path (default: <shard_dir>/pytorch_model.safetensors)",
    )
    return parser.parse_args()


# =============================
# Step 1: Load shards
# =============================
def load_model_shards(shard_dir):
    shards = []
    for f in sorted(os.listdir(shard_dir)):
        if f.startswith("model_world_size") and f.endswith(".pt"):
            print("Loading shard:", f)
            # weights_only=False, because FSDP shards contain DTensor / bucket info
            shard = torch.load(
                os.path.join(shard_dir, f),
                map_location="cpu",
                weights_only=False,
            )
            shards.append(shard)
    return shards


# =============================
# Step 2: Merge shards
# =============================
def merge_fsdp_shards(shards):
    tensor_parts = defaultdict(list)
    meta_entries = {}
    for shard in shards:
        # Checkpoint usually contains {'model': {...}, 'optimizer': {...}, ...}
        if isinstance(shard, dict):
            if "model" in shard and isinstance(shard["model"], dict):
                state_dict = shard["model"]
            elif "state_dict" in shard and isinstance(shard["state_dict"], dict):
                state_dict = shard["state_dict"]
            else:
                state_dict = shard
        else:
            state_dict = shard

        for k, v in state_dict.items():
            # Non-tensor (e.g., meta info) keep one copy
            if not isinstance(v, torch.Tensor):
                if k not in meta_entries:
                    meta_entries[k] = v
                continue

            # If still DTensor, convert to local tensor first
            if v.__class__.__name__ == "DTensor" or hasattr(v, "to_local"):
                try:
                    v = v.to_local()
                except Exception:
                    pass

            # detach + cpu + contiguous + clone, try to detach from original FSDP storage
            v = v.detach().to("cpu").contiguous().clone()

            # Additional storage validity check to avoid safetensors crash later
            try:
                if hasattr(v, "untyped_storage"):
                    _ = v.untyped_storage().data_ptr()
                else:
                    _ = v.storage().data_ptr()
            except RuntimeError as e:
                print(f"[WARN] Skip tensor '{k}' in merge due to invalid storage: {e}")
                continue

            tensor_parts[k].append(v)

    merged = {}
    for k, parts in tensor_parts.items():
        if len(parts) == 1:
            merged[k] = parts[0]
        else:
            try:
                merged[k] = torch.cat(parts, dim=0)
            except Exception as e:
                print(
                    f"[WARN] Failed to concat shards for {k}: {e}. Using first chunk only."
                )
                merged[k] = parts[0]

    merged.update(meta_entries)
    return merged


# =============================
# Step 3: Save safetensors (with invalid storage filtering)
# =============================
def save_safetensors(merged, path):
    """
    Filter out "valid storage" tensors from merged and write to safetensors.
    Tensors with invalid storage will be skipped and a warning will be printed.
    """
    tensor_state_dict = {}
    skipped = []

    for k, v in merged.items():
        if not isinstance(v, torch.Tensor):
            continue

        # Simulate storage_ptr access logic in safetensors, filter in advance
        try:
            if hasattr(v, "untyped_storage"):
                _ = v.untyped_storage().data_ptr()
            else:
                _ = v.storage().data_ptr()
        except RuntimeError as e:
            print(f"[WARN] Skip tensor '{k}' in save due to invalid storage: {e}")
            skipped.append(k)
            continue

        tensor_state_dict[k] = v

    print(
        f"Saving {len(tensor_state_dict)} tensors to {path} "
        f"(skipped {len(skipped)} tensors with invalid storage)..."
    )
    if skipped:
        print("Skipped keys (likely optimizer/FSDP internal states etc.):")
        for name in skipped:
            print("  -", name)

    save_file(tensor_state_dict, path)
    print("Saved successfully!")


if __name__ == "__main__":
    args = parse_args()
    output = args.output or os.path.join(args.shard_dir, "pytorch_model.safetensors")
    shards = load_model_shards(args.shard_dir)
    save_safetensors(merge_fsdp_shards(shards), output)
