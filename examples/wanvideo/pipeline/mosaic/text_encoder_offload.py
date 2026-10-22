"""Keep the UMT5-XXL text encoder off the GPU between prompt encodings.

The encoder holds ~11.4 GB of weights but each prompt embedding is only
512 x 4096 values (~4 MB), and inference encodes the same caption and
negative prompt over and over. With ``MG35_T5_OFFLOAD=1`` the encoder lives
on CPU; a cache miss moves it to the GPU just for that call, then back, and
hits return the stored embedding. Outputs are numerically identical (same
weights, device and dtype for the computation), and the ~11 GB this frees
lets multi-block generation fit on 40 GB GPUs, where it otherwise runs out
of memory during DA3 memory registration.

Both call sites (``WanVideoUnit_PromptEmbedder`` and the distilled rollout's
``_enc``) call ``pipe.text_encoder(ids, mask)``, so swapping in the wrapper
needs no other change.
"""

import hashlib
import os
from collections import OrderedDict

import torch

ENV_VAR = "MG35_T5_OFFLOAD"


class OffloadedTextEncoder(torch.nn.Module):
    def __init__(self, encoder, max_entries=64):
        super().__init__()
        # Stored outside _modules so module.to(device) / accelerator.prepare
        # never move the weights back onto the GPU.
        self.__dict__["_encoder"] = encoder.to("cpu")
        self._cache = OrderedDict()
        self._max_entries = int(max_entries)
        self.misses = 0

    @staticmethod
    def _key(ids, mask):
        digest = hashlib.sha1()
        for t in (ids, mask):
            t = t.detach().cpu().contiguous()
            digest.update(str((tuple(t.shape), str(t.dtype))).encode())
            digest.update(t.numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def forward(self, ids, mask):
        key = self._key(ids, mask)
        emb = self._cache.get(key)
        if emb is None:
            self.misses += 1
            encoder = self.__dict__["_encoder"]
            device = ids.device
            encoder.to(device)
            try:
                emb = encoder(ids, mask).detach().to("cpu")
            finally:
                encoder.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            self._cache[key] = emb
            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        # callers zero padding positions in place; never hand out the cache
        return emb.to(ids.device).clone()


def maybe_offload_text_encoder(pipe, *, verbose=True):
    """Install the wrapper on ``pipe`` when MG35_T5_OFFLOAD is truthy."""
    if str(os.environ.get(ENV_VAR, "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    encoder = getattr(pipe, "text_encoder", None)
    if encoder is None or isinstance(encoder, OffloadedTextEncoder):
        return False
    pipe.text_encoder = OffloadedTextEncoder(encoder)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(
            f"[{ENV_VAR}] text encoder moved to CPU; prompt embeddings are "
            "computed on GPU on first use and cached",
            flush=True,
        )
    return True
