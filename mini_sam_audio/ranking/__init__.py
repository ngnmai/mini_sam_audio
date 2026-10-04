# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from mini_sam_audio.model.config import (
    EnsembleRankerConfig,
    ImageBindRankerConfig,
    JudgeRankerConfig,
    SoundActivityRankerConfig,
)
from mini_sam_audio.ranking.imagebind import ImageBindRanker
from mini_sam_audio.ranking.judge import JudgeRanker
from mini_sam_audio.ranking.ranker import EnsembleRanker
from mini_sam_audio.ranking.sound_activity import SoundActivityRanker


def create_ranker(config):
    if isinstance(config, ImageBindRankerConfig):
        return ImageBindRanker(config)
    elif isinstance(config, JudgeRankerConfig):
        return JudgeRanker(config)
    elif isinstance(config, SoundActivityRankerConfig):
        return SoundActivityRanker(config)
    elif isinstance(config, EnsembleRankerConfig):
        ranker_cfgs, weights = zip(*config.rankers.values(), strict=False)
        return EnsembleRanker(
            rankers=[create_ranker(cfg) for cfg in ranker_cfgs],
            weights=weights,
        )
    else:
        assert config is None
        return None
