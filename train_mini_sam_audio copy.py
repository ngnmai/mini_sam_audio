"""Train Mini SAM-Audio with a custom waveform/STFT/alignment/mixture-consistency loss.

Uses PyTorch Lightning with the DDP strategy so the same entry point scales from a
single GPU to multi-node jobs on CSC's Roihu (launched via ``srun``/``sbatch``,
which Lightning's SLURM environment auto-detects).

Reads soft-label supervised training data from ``--train-data-root``/``--val-data-root`` using
the same directory layout as ``utils/generate_soft_labels.py``::

    data_root/audio/<stem>.wav
    data_root/video/<stem>.mp4
    data_root/mask/<stem>.mp4
    data_root/soft_labels/<stem>.wav
    data_root/residual_labels/<stem>.wav  (optional)

Example (single node):
    python train_mini_sam_audio.py --train-data-root /path/to/train --val-data-root /path/to/val \\
        --devices 4

Example (Roihu, multi-node, launched per-task via srun):
    srun python train_mini_sam_audio.py --train-data-root /path/to/train --val-data-root /path/to/val \\
        --devices 4 --num-nodes 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import mlflow
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_sam_audio import MiniSAMAudio
from mini_sam_audio.compression.model_loading import bootstrap_model_and_processor

from mini_sam_audio.model.loss import (
    MiniSAMAudioLoss,
    SeparationPredictions,
    SeparationTargets,
    compute_si_sdr_db,
)

from utils.dataset_soft_labels import CollatedBatch, SoftLabelDataset, make_collate_fn


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
DEFAULT_MLFLOW_EXPERIMENT_NAME = "mini_sam_audio"
DEFAULT_LR_SCHEDULER = "step"
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_LR_DECAY_STEP_EPOCHS = 10
DEFAULT_LR_DECAY_GAMMA = 0.5


class MiniSAMAudioLightningModule(L.LightningModule):
    def __init__(
        self,
        model: MiniSAMAudio,
        lr: float,
        w_wave_l1: float,
        w_stft_l1: float,
        w_align: float,
        w_mix_consistency: float,
        freeze_backbones: bool,
        lr_scheduler: str = "none",
        warmup_ratio: float = DEFAULT_WARMUP_RATIO,
        lr_decay_step_epochs: int = DEFAULT_LR_DECAY_STEP_EPOCHS,
        lr_decay_gamma: float = DEFAULT_LR_DECAY_GAMMA,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        if freeze_backbones:
            self.model.freeze_inference_backbones()
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
        print("transferring batch to device", flush=True)
        batch.batch = batch.batch.to(device)
        batch.target_wav = batch.target_wav.to(device)
        batch.residual_wav = batch.residual_wav.to(device)
        return batch

    def _shared_step(self, collated: CollatedBatch, stage: str, batch_idx: int) -> torch.Tensor:
        try:
            print("output of model.separate_for_training", flush=True)
            outputs = self.model.separate_for_training(collated.batch)

            print("separate output obtained", flush=True)
            preds = SeparationPredictions(
                pred_target_wav=outputs.pred_target_wav,
                pred_residual_wav=outputs.pred_residual_wav,
                pred_target_latent_128=outputs.pred_target_latent_128,
                video_embed_128=outputs.video_embed_128,
                audio_pad_mask=outputs.audio_pad_mask,
                wav_sizes=outputs.wav_sizes,
            )
            print("set up targets", flush=True)
            targets = SeparationTargets(
                mixture_wav=collated.batch.audios.squeeze(1),
                target_wav=collated.target_wav,
                residual_wav=collated.residual_wav,
            )

            print("set up losses", flush=True)
            losses = self.loss_fn(preds, targets)
            batch_size = collated.batch.audios.size(0)
            # CHANGE BEGINS
            # Decoded wavs can differ in length from the soft-label targets; crop both
            # to their shared length so si_sdr_db sees matching shapes.
            si_sdr_t_wav = min(outputs.pred_target_wav.size(-1), collated.target_wav.size(-1))
            si_sdr_wav_sizes = (
                outputs.wav_sizes.clamp(max=si_sdr_t_wav)
                if outputs.wav_sizes is not None
                else None
            )
            si_sdr_db = compute_si_sdr_db(
                outputs.pred_target_wav[..., :si_sdr_t_wav],
                collated.target_wav[..., :si_sdr_t_wav],
                si_sdr_wav_sizes,
            ).mean()
            # CHANGE ENDS
            print("computed si_sdr_db -> DONE", flush=True)

            for name, value in losses.items():
                metric_name = f"{stage}_{name}"
                self._log_metric(
                    metric_name,
                    value,
                    batch_size=batch_size,
                    prog_bar=metric_name.endswith("loss_total") or metric_name.endswith("si_sdr_db"),
                )
            print("logged all metrics", flush=True)
            self._log_metric(f"{stage}_si_sdr_db", si_sdr_db, batch_size=batch_size, prog_bar=True)
        except Exception as exc:
            print(
                f"[train_mini_sam_audio] _shared_step failed: stage={stage} batch_idx={batch_idx} "
                f"global_step={self.global_step} global_rank={self.trainer.global_rank} "
                f"audios.shape={collated.batch.audios.shape} target_wav.shape={collated.target_wav.shape} "
                f"error={exc}",
                flush=True,
            )
            raise
        return losses["loss_total"]

    def _log_metric(self, name: str, value: torch.Tensor, batch_size: int, prog_bar: bool) -> None:
        self.log(
            name,
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=True,
        )

    def training_step(self, collated: CollatedBatch, batch_idx: int) -> torch.Tensor:
        print("starting training step", flush=True)
        return self._shared_step(collated, stage="train", batch_idx=batch_idx)

    def validation_step(self, collated: CollatedBatch, batch_idx: int) -> torch.Tensor:
        print("starting validation step", flush=True)
        return self._shared_step(collated, stage="val", batch_idx=batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        forbidden_prefixes = ("audio_codec.", "vision_encoder.")
        named_trainable_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        forbidden_trainable = [
            name
            for name, _ in named_trainable_params
            if name.startswith(forbidden_prefixes)
        ]
        if forbidden_trainable:
            joined = ", ".join(forbidden_trainable[:5])
            raise RuntimeError(
                f"Frozen-backbone parameters unexpectedly trainable: {joined}"
            )

        trainable_params = [param for _, param in named_trainable_params]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer setup")

        print("configured optimizer with trainable parameters", flush=True)

        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.lr)
        if self.hparams.lr_scheduler == "none":
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(total_steps // max(self.trainer.max_epochs, 1), 1)
        warmup_steps = int(round(self.hparams.warmup_ratio * total_steps))

        if self.hparams.lr_scheduler == "step":
            decay_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(self.hparams.lr_decay_step_epochs * steps_per_epoch, 1),
                gamma=self.hparams.lr_decay_gamma,
            )
        elif self.hparams.lr_scheduler == "cosine":
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1)
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {self.hparams.lr_scheduler!r}")

        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = decay_scheduler

        print(
            f"[train_mini_sam_audio] LR schedule: {self.hparams.lr_scheduler} "
            f"warmup_steps={warmup_steps} total_steps={total_steps}",
            flush=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _parse_devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Mini SAM-Audio using a weighted waveform/STFT/alignment/mixture-consistency "
            "loss against soft-label supervision, via PyTorch Lightning DDP."
        )
    )
    parser.add_argument(
        "--train-data-root",
        type=Path,
        required=True,
        help="Directory containing audio/, video/, mask/, and soft_labels/ subfolders for training.",
    )
    parser.add_argument(
        "--val-data-root",
        type=Path,
        required=True,
        help="Directory with the same layout as --train-data-root, used for validation.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["scratch", "checkpoint"],
        default="scratch",
        help=(
            "Model/processor bootstrap mode. 'scratch' builds a fresh model/processor from local "
            "JSON configs with no network access. 'checkpoint' loads weights from --checkpoint-path."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local SAM-Audio checkpoint path. Required when --init-mode=checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=(
            "Local JSON file with MiniSAMAudioConfig fields. Optional in scratch mode (falls back to "
            "built-in defaults) and checkpoint mode (falls back to config.json next to --checkpoint-path)."
        ),
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        default=None,
        help=(
            "Local JSON file with 'audio_hop_length' and 'audio_sampling_rate' fields for "
            "MiniSAMAudioProcessor. Optional; falls back to values derived from the model config."
        ),
    )
    parser.add_argument(
        "--unfreeze-inference-backbones",
        action="store_true",
        default=False,
        help=(
            "Keep audio_codec/vision_encoder trainable in scratch mode instead of the default "
            "frozen behavior. Checkpoint mode always freezes them regardless of this flag, since "
            "both backbones are pretrained via their installed packages (dacvae, perception-models) "
            "and only the rest of the model trains from scratch."
        ),
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
        "--lr-scheduler",
        type=str,
        choices=["none", "step", "cosine"],
        default=DEFAULT_LR_SCHEDULER,
        help="LR schedule applied after warmup: 'none' keeps --lr constant.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Fraction of total training steps spent linearly warming up the LR.",
    )
    parser.add_argument(
        "--lr-decay-step-epochs",
        type=int,
        default=DEFAULT_LR_DECAY_STEP_EPOCHS,
        help="Epochs between LR decay steps when --lr-scheduler=step.",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=DEFAULT_LR_DECAY_GAMMA,
        help="Multiplicative LR decay factor when --lr-scheduler=step.",
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
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        default=False,
        help="Enable PyTorch autograd anomaly detection for debugging NaN/Inf gradients.",
    )
    parser.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=2,
        help="Number of validation sanity-check batches to run before training starts.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "Optional MLflow tracking URI override (directory or sqlite:/// path). When unset, uses "
            "whatever tracking URI is already configured in the environment (e.g. MLFLOW_TRACKING_URI)."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name. Defaults to the SLURM_JOB_ID environment variable when unset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Build the model/processor, print a summary, and exit without starting training.",
    )
    args = parser.parse_args()
    if args.init_mode == "checkpoint" and args.checkpoint_path is None:
        parser.error("--checkpoint-path is required when --init-mode=checkpoint.")
    return args


def build_output_dir(train_data_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return train_data_root / "checkpoints"


def print_model_summary(model: MiniSAMAudio, batch_size: int) -> None:
    """Print the module tree and, when available, a torchinfo parameter/shape summary."""
    print(model)
    print()
    try:
        from torchinfo import summary
    except ImportError:
        print("[train_mini_sam_audio] torchinfo not installed; skipping tensor-shape summary.")
        return

    try:
        print(summary(model, depth=4, verbose=0, row_settings=["var_names"]))
    except Exception as exc:  # torchinfo requires a concrete input batch to trace shapes; skip gracefully.
        print(f"[train_mini_sam_audio] torchinfo summary unavailable ({exc}); printed module tree only.")


def main(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    print(
        f"[train_mini_sam_audio] RANK={os.environ.get('RANK', 'unset')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', 'unset')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 'unset')} "
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'unset')} "
        f"SLURM_NODELIST={os.environ.get('SLURM_NODELIST', 'unset')}",
        flush=True,
    )
    mlflow.autolog()

    output_dir = build_output_dir(args.train_data_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[train_mini_sam_audio] Bootstrapping model and processor...", flush=True)
    model, processor, freeze_backbones = bootstrap_model_and_processor(
        init_mode=args.init_mode,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config,
        processor_config_path=args.processor_config,
        unfreeze_inference_backbones=args.unfreeze_inference_backbones,
    )
    print(
        f"[train_mini_sam_audio] init_mode={args.init_mode} "
        f"freeze_inference_backbones={freeze_backbones}",
        flush=True,
    )

    if args.dry_run:
        if freeze_backbones:
            model.freeze_inference_backbones()
        print_model_summary(model, batch_size=args.batch_size)
        print("[train_mini_sam_audio] --dry-run set; exiting before dataset/trainer setup.")
        return

    print(f"[train_mini_sam_audio] Loading train dataset from {args.train_data_root}...", flush=True)
    train_dataset = SoftLabelDataset(args.train_data_root)
    print(f"[train_mini_sam_audio] Train dataset loaded: {len(train_dataset)} samples", flush=True)

    print(f"[train_mini_sam_audio] Loading val dataset from {args.val_data_root}...", flush=True)
    val_dataset = SoftLabelDataset(args.val_data_root)
    print(f"[train_mini_sam_audio] Val dataset loaded: {len(val_dataset)} samples", flush=True)

    collate_fn = make_collate_fn(processor, processor.audio_sampling_rate)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    print(
        f"[train_mini_sam_audio] Dataloaders ready: "
        f"{len(train_dataloader)} train batches, {len(val_dataloader)} val batches",
        flush=True,
    )

    print("[train_mini_sam_audio] Building Lightning module...", flush=True)
    module = MiniSAMAudioLightningModule(
        model=model,
        lr=args.lr,
        w_wave_l1=args.w_wave_l1,
        w_stft_l1=args.w_stft_l1,
        w_align=args.w_align,
        w_mix_consistency=args.w_mix_consistency,
        freeze_backbones=freeze_backbones,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        lr_decay_step_epochs=args.lr_decay_step_epochs,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    mlflow_run_name = args.mlflow_run_name or os.environ.get("SLURM_JOB_ID")
    print(
        f"[train_mini_sam_audio] Configuring MLflow tracking "
        f"(tracking_uri={args.mlflow_tracking_uri or '<environment-configured>'}, "
        f"run_name={mlflow_run_name})...",
        flush=True,
    )
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision="bf16-mixed",
        logger=False,
        strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=args.log_every,
        detect_anomaly=args.detect_anomaly,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )

    if trainer.is_global_zero:
        mlflow.start_run(run_name=mlflow_run_name)
        mlflow.log_params(
            {key: (str(value) if value is not None else "None") for key, value in vars(args).items()}
        )

    print("[train_mini_sam_audio] Starting training...", flush=True)
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if trainer.is_global_zero:
        checkpoint_path = output_dir / "mini_sam_audio_final.pt"
        torch.save(module.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.end_run()


if __name__ == "__main__":
    main(parse_args())
