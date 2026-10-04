"""Training loss module for mini_sam_audio separation outputs."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SeparationPredictions:
    pred_target_wav: torch.Tensor  # [B, T_wav]
    pred_residual_wav: torch.Tensor  # [B, T_wav]
    pred_target_latent_128: torch.Tensor  # [B, 128, T_feat]
    video_embed_128: Optional[torch.Tensor]  # [B, 128, T_feat]
    audio_pad_mask: Optional[torch.Tensor]  # [B, T_feat]
    wav_sizes: Optional[torch.Tensor]  # [B]


@dataclass
class SeparationTargets:
    mixture_wav: torch.Tensor  # [B, T_wav]
    target_wav: torch.Tensor  # [B, T_wav]
    residual_wav: Optional[torch.Tensor]  # [B, T_wav]


class MiniSAMAudioLoss(torch.nn.Module):
    def __init__(
        self,
        w_wave_l1: float = 1.0,
        w_stft_l1: float = 1.0,
        w_align: float = 0.1,
        w_mix_consistency: float = 0.5,
        stft_n_fft: int = 1024,
        stft_hop_length: int = 256,
        stft_win_length: int = 1024,
        stft_window: str = "hann",
        eps: float = 1e-8,
    ):
        super().__init__()
        if stft_window != "hann":
            raise NotImplementedError(f"Unsupported stft_window: {stft_window!r}")
        self.w_wave_l1 = w_wave_l1
        self.w_stft_l1 = w_stft_l1
        self.w_align = w_align
        self.w_mix_consistency = w_mix_consistency
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.eps = eps
        self.register_buffer(
            "_stft_window", torch.hann_window(stft_win_length), persistent=False
        )

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        # [B, max_len] bool mask, True where index is within the valid length
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def _masked_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        diff = (pred - target).abs()
        if mask is None:
            return diff.mean()
        # grow mask with singleton dims (e.g. frequency axis) so it broadcasts over diff
        while mask.dim() < diff.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(diff).to(diff.dtype)
        return (diff * mask).sum() / mask.sum().clamp_min(self.eps)

    def _stft_mag(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=self._stft_window,
            center=True,
            return_complex=True,
        )
        return spec.abs()  # [B, F, T_stft]

    def _alignment_loss(
        self,
        pred_latent: torch.Tensor,
        video_embed: torch.Tensor,
        audio_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # center/normalize across the channel dim to get a per (batch, time) correlation
        pred_centered = pred_latent - pred_latent.mean(dim=1, keepdim=True)
        video_centered = video_embed - video_embed.mean(dim=1, keepdim=True)
        pred_std = pred_centered.std(dim=1)
        video_std = video_centered.std(dim=1)
        # eps avoids div-by-zero when a channel vector is near-constant
        corr = (pred_centered * video_centered).mean(dim=1) / (
            pred_std * video_std + self.eps
        )
        dist = 1 - corr  # [B, T_feat]
        if audio_pad_mask is None:
            return dist.mean()
        mask = audio_pad_mask.to(dist.dtype)
        return (dist * mask).sum() / mask.sum().clamp_min(self.eps)

    def forward(
        self,
        preds: SeparationPredictions,
        targets: SeparationTargets,
    ) -> dict[str, torch.Tensor]:
        t_wav = preds.pred_target_wav.size(-1)
        wav_mask = None
        if preds.wav_sizes is not None:
            wav_mask = self._lengths_to_mask(preds.wav_sizes, t_wav)

        loss_wave_l1 = self._masked_l1(
            preds.pred_target_wav, targets.target_wav, wav_mask
        )
        if targets.residual_wav is not None:
            loss_wave_l1 = loss_wave_l1 + self._masked_l1(
                preds.pred_residual_wav, targets.residual_wav, wav_mask
            )

        pred_target_mag = self._stft_mag(preds.pred_target_wav)
        target_mag = self._stft_mag(targets.target_wav)
        stft_mask = None
        if preds.wav_sizes is not None:
            stft_lengths = (
                torch.div(preds.wav_sizes, self.stft_hop_length, rounding_mode="floor")
                + 1
            ).clamp(max=pred_target_mag.size(-1))
            stft_mask = self._lengths_to_mask(stft_lengths, pred_target_mag.size(-1))
        loss_stft_l1 = self._masked_l1(pred_target_mag, target_mag, stft_mask)
        if targets.residual_wav is not None:
            pred_residual_mag = self._stft_mag(preds.pred_residual_wav)
            residual_target_mag = self._stft_mag(targets.residual_wav)
            loss_stft_l1 = loss_stft_l1 + self._masked_l1(
                pred_residual_mag, residual_target_mag, stft_mask
            )

        if preds.video_embed_128 is not None:
            loss_align = self._alignment_loss(
                preds.pred_target_latent_128,
                preds.video_embed_128,
                preds.audio_pad_mask,
            )
        else:
            loss_align = torch.zeros(
                (),
                device=preds.pred_target_latent_128.device,
                dtype=preds.pred_target_latent_128.dtype,
            )

        mix_pred = preds.pred_target_wav + preds.pred_residual_wav
        loss_mix_consistency = self._masked_l1(mix_pred, targets.mixture_wav, wav_mask)

        loss_total = (
            self.w_wave_l1 * loss_wave_l1
            + self.w_stft_l1 * loss_stft_l1
            + self.w_align * loss_align
            + self.w_mix_consistency * loss_mix_consistency
        )

        return {
            "loss_total": loss_total,
            "loss_wave_l1": loss_wave_l1,
            "loss_stft_l1": loss_stft_l1,
            "loss_align": loss_align,
            "loss_mix_consistency": loss_mix_consistency,
        }


__all__ = ["SeparationPredictions", "SeparationTargets", "MiniSAMAudioLoss"]
