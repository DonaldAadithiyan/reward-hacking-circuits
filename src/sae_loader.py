"""Lazy loader for the pretrained SAEs, one per (layer, submodule).

Memory note (M4, 16GB): SAEs are ~75-100MB each on CPU. We keep them cached but
they can be dropped. During attribution we hold the active set; the TASK warns not
to hold all in memory but 3 families x 12 layers of GPT-2 SAEs (~2GB) fits in 16GB.
"""
from __future__ import annotations
import functools
from sae_lens import SAE
import config as C

def hook_name(layer: int, submodule: str) -> str:
    return {
        "resid": f"blocks.{layer}.hook_resid_pre",
        "attn":  f"blocks.{layer}.hook_attn_out",
        "mlp":   f"blocks.{layer}.hook_mlp_out",
    }[submodule]

@functools.lru_cache(maxsize=None)
def get_sae(layer: int, submodule: str, device: str | None = None):
    device = device or C.DEVICE
    release, sae_id_tmpl, _ = C.SAE_RELEASES[submodule]
    sae_id = sae_id_tmpl.format(l=layer)
    sae = SAE.from_pretrained(release, sae_id, device=device)
    sae = sae[0] if isinstance(sae, tuple) else sae
    sae = sae.to(device)
    for p in sae.parameters():
        p.requires_grad_(False)
    sae.eval()
    return sae

def all_nodes():
    """All (layer, submodule) submodule slots we attribute over."""
    return [(l, s) for l in C.SAE_LAYERS for s in C.SUBMODULES]
