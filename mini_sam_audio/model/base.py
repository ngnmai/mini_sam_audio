# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Callable

import torch


class BaseModel(torch.nn.Module):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_config(cls, config=None, **kwargs):
        """Instantiate a model from a config object or config kwargs."""
        if config is None:
            config = cls.config_cls(**kwargs)
        elif isinstance(config, dict):
            config = cls.config_cls(**config)
        return cls(config)
