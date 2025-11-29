# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from https://github.com/facebookresearch/XLM/blob/main/xlm/model/memory/__init__.py


from .memory import HashingMemory, ProductKeyArgs
from .xformer_embeddingbag import xFormerEmbeddingBag

__all__ = ['HashingMemory', 'ProductKeyArgs', 'xFormerEmbeddingBag']