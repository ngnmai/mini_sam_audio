"""Train Mini SAMAudio with a custom waveform/STFT/alignment/mixture-consistency loss.

Uses PyTorch Lightning with the DDP strategy so the same entry point scales from a
single GPU to multi-node jobs on CSC's Roihu (launched via ``srun``/``sbatch``,
which Lightning's SLURM environment auto-detects).

Reads soft-label supervised training data from ``--data-root`` using the same
directory layout as ``utils/generate_soft_labels.py``::

    data_root/audio/<stem>.wav
    data_root/video/<stem>.mp4
    data_root/mask/<stem>.mp4
    data_root/soft_labels/<stem>.wav
    data_root/residual_labels/<stem>.wav  (optional)

Example (single node):
    python train_mini_sam_audio.py --data-root /path/to/data_root --devices 4

Example (Roihu, multi-node, launched per-task via srun):
    srun python train_mini_sam_audio.py --data-root /path/to/data_root \\
        --devices 4 --num-nodes 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_sam_audio import MiniSAMAudio, SAMAudioProcessor

from mini_sam_audio.model.loss import (
    MiniSAMAudioLoss,
    SeparationPredictions,
    SeparationTargets,
)

from utils.dataset_soft_labels import CollatedBatch, SoftLabelDataset, make_collate_fn


DEFAULT_CHECKPOINT_PATH = "facebook/sam-audio-small"
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_EPOCHS = 1
DEFAULT_LR = 1e-4
DEFAULT_W_WAVE_L1 = 1.0
DEFAULT_W_STFT_L1 = 1.0
DEFAULT_W_ALIGN = 0.1
DEFAULT_W_MIX_CONSISTENCY = 0.5
DEFAULT_LOG_EVERY = 10
DEFAULT_NUM_NODES = 1
DEFAULT_DEVICES = "auto"


class MiniSAMAudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        lr: float,
        w_wave_l1: float,
        w_stft_l1: float,
        w_align: float,
        w_mix_consistency: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MiniSAMAudio.from_pretrained(checkpoint_path)
        self.loss_fn = MiniSAMAudioLoss(
            w_wave_l1=w_wave_l1,
            w_stft_l1=w_stft_l1,
            w_align=w_align,
            w_mix_consistency=w_mix_consistency,
        )

    def transfer_batch_to_device(
        self, batch: CollatedBatch, device: torch.device, dataloader_idx: int
    ) -> CollatedBatch:
        # CollatedBatch wraps a custom `Batch` type Lightning doesn't move automatically.
        batch.batch = batch.batch.to(device)
        batch.target_wav = batch.target_wav.to(device)
        batch.residual_wav = batch.residual_wav.to(device)
        return batch

    def training_step(self, collated: CollatedBatch, batch_idx: int) -> torch.Tensor:
        outputs = self.model.separate_for_training(collated.batch)

        # preds
        preds = SeparationPredictions(
            pred_target_wav=outputs.pred_target_wav,
            pred_residual_wav=outputs.pred_residual_wav,
            pred_target_latent_128=outputs.pred_target_latent_128,
            video_embed_128=outputs.video_embed_128,
            audio_pad_mask=outputs.audio_pad_mask,
            wav_sizes=outputs.wav_sizes,
        )

        # ground truth
        targets = SeparationTargets(
            mixture_wav=collated.batch.audios.squeeze(1),
            target_wav=collated.target_wav,
            residual_wav=collated.residual_wav,
        )

        losses = self.loss_fn(preds, targets)
        batch_size = collated.batch.audios.size(0)
        for name, value in losses.items():
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(name == "loss_total"),
                batch_size=batch_size,
                sync_dist=True,
            )
        return losses["loss_total"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def _parse_devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Mini SAMAudio using a weighted waveform/STFT/alignment/mixture-consistency "
            "loss against soft-label supervision, via PyTorch Lightning DDP."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing audio/, video/, mask/, and soft_labels/ subfolders.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="SAM-Audio checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write checkpoints and Lightning logs. Defaults to <data-root>/checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device training batch size.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker processes.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam learning rate.")
    parser.add_argument("--w-wave-l1", type=float, default=DEFAULT_W_WAVE_L1, help="Waveform L1 loss weight.")
    parser.add_argument("--w-stft-l1", type=float, default=DEFAULT_W_STFT_L1, help="STFT magnitude L1 loss weight.")
    parser.add_argument("--w-align", type=float, default=DEFAULT_W_ALIGN, help="Audio/video alignment loss weight.")
    parser.add_argument(
        "--w-mix-consistency",
        type=float,
        default=DEFAULT_W_MIX_CONSISTENCY,
        help="Mixture-consistency loss weight.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu" if torch.cuda.is_available() else "cpu",
        help="Lightning accelerator (gpu, cpu, auto).",
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=DEFAULT_DEVICES,
        help="Devices per node: an int count, explicit list, or 'auto'.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=DEFAULT_NUM_NODES,
        help="Number of nodes for multi-node DDP (set to match the Slurm job on Roihu).",
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help="Log loss values every N steps.")
    return parser.parse_args()


def build_output_dir(data_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return data_root / "checkpoints"


def main(args: argparse.Namespace) -> None:
    output_dir = build_output_dir(args.data_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    dataset = SoftLabelDataset(args.data_root)
    collate_fn = make_collate_fn(processor, processor.audio_sampling_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    module = MiniSAMAudioLightningModule(
        checkpoint_path=args.checkpoint_path,
        lr=args.lr,
        w_wave_l1=args.w_wave_l1,
        w_stft_l1=args.w_stft_l1,
        w_align=args.w_align,
        w_mix_consistency=args.w_mix_consistency,
    )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        # find_unused_parameters: the frozen vision encoder runs under no_grad and never gets gradients.
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=args.log_every,
    )
    trainer.fit(module, train_dataloaders=dataloader)

    if trainer.is_global_zero:
        checkpoint_path = output_dir / "mini_sam_audio_final.pt"
        torch.save(module.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main(parse_args())
