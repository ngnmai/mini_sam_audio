# System libs
import os
import random
import time
from pathlib import Path

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from PIL import Image
import clip
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from modules import models
from utils import AverageMeter, istft_reconstruction, warpgrid
from dataset import video_transforms as vtransforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_, clip_preprocess = clip.load("ViT-B/32", device="cpu")


class VGGDirMixDataset(torch.utils.data.Dataset):
    def __init__(self, args, split="test"):
        self.split = split
        self.num_mix = args.num_mix
        self.num_frames = args.num_frames
        self.stride_frames = args.stride_frames
        self.fps = args.frameRate
        self.imgSize = args.imgSize
        self.audRate = args.audRate
        self.audLen = args.audLen
        self.audSec = 1.0 * self.audLen / self.audRate
        self.log_freq = args.log_freq
        self.stft_frame = args.stft_frame
        self.stft_hop = args.stft_hop
        self.seed = args.seed
        self.arch_frame = args.arch_frame

        self.audio_dir = Path(args.audio_dir).expanduser().resolve()
        self.video_dir = Path(args.video_dir).expanduser().resolve()
        self.audio_exts = tuple(
            ext.strip().lower() for ext in args.audio_exts.split(",") if ext.strip()
        )

        self._init_vtransform()
        self.samples = self._index_samples()
        assert len(self.samples) > 1, "Need at least 2 matched audio/video samples"
        print("# matched VGG samples: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def _init_vtransform(self):
        transform_list = [
            vtransforms.Resize(self.imgSize, InterpolationMode.BICUBIC),
            vtransforms.CenterCrop(self.imgSize),
            vtransforms.ToTensor(),
            vtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            vtransforms.Stack(),
        ]
        self.vid_transform = transforms.Compose(transform_list)
        self.clip_transform = transforms.Compose([vtransforms.Stack()])

    def _index_samples(self):
        audio_map = {}
        for p in self.audio_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.audio_exts:
                continue
            rel = p.relative_to(self.audio_dir)
            key = str(rel.with_suffix(""))
            audio_map[key] = p

        samples = []
        for key, audio_path in audio_map.items():
            frame_dir = self.video_dir / key
            if not frame_dir.is_dir():
                continue
            frame_paths = sorted(
                [
                    p
                    for p in frame_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
            )
            if len(frame_paths) < 1:
                continue

            cls = Path(key).parts[0] if len(Path(key).parts) > 0 else "unknown"
            samples.append(
                {
                    "key": key,
                    "class": cls,
                    "audio": str(audio_path),
                    "frame_dir": str(frame_dir),
                    "frames": [str(x) for x in frame_paths],
                }
            )

        return sorted(samples, key=lambda x: x["key"])

    def _load_frames(self, paths):
        frames = [Image.open(path).convert("RGB") for path in paths]
        return self.vid_transform(frames)

    def _load_frames_clip(self, paths):
        frames = [clip_preprocess(Image.open(path).convert("RGB")) for path in paths]
        return self.clip_transform(frames)

    def _stft(self, audio):
        spec = librosa.stft(audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio(self, path, center_timestamp):
        audio = np.zeros(self.audLen, dtype=np.float32)
        audio_raw, rate = librosa.load(path, sr=None, mono=True)

        if rate > self.audRate:
            audio_raw = librosa.resample(audio_raw, orig_sr=rate, target_sr=self.audRate)
            rate = self.audRate
        elif rate < self.audRate:
            audio_raw = librosa.resample(audio_raw, orig_sr=rate, target_sr=self.audRate)
            rate = self.audRate

        if audio_raw.shape[0] < self.audRate * self.audSec:
            n = int(self.audRate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)
        audio[self.audLen // 2 - (center - start): self.audLen // 2 + (end - center)] = audio_raw[start:end]

        audio[audio > 1.0] = 1.0
        audio[audio < -1.0] = -1.0
        return audio

    def _mix_n_and_stft(self, audios):
        n_src = len(audios)
        mags = [None for _ in range(n_src)]
        audio_mix = 0
        for n in range(n_src):
            audio_mix += audios[n]
        audio_mix /= n_src

        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(n_src):
            amp_n, _ = self._stft(audios[n])
            mags[n] = amp_n.unsqueeze(0)
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0), audio_mix

    def _pick_indices(self, index):
        indices = [index]
        used_classes = {self.samples[index]["class"]}
        step = 1
        while len(indices) < self.num_mix:
            cand = (index + step) % len(self.samples)
            step += 1
            cls = self.samples[cand]["class"]
            if cls in used_classes:
                continue
            indices.append(cand)
            used_classes.add(cls)

        return indices

    def _pick_frame_window(self, frame_paths):
        total = len(frame_paths)
        center = total // 2
        paths = []
        for i in range(self.num_frames):
            offset = (i - self.num_frames // 2) * self.stride_frames
            idx = center + offset
            idx = max(0, min(total - 1, idx))
            paths.append(frame_paths[idx])
        return paths, center

    def __getitem__(self, index):
        sel = self._pick_indices(index)
        n_src = self.num_mix
        frames = [None for _ in range(n_src)]
        audios = [None for _ in range(n_src)]
        infos = [[] for _ in range(n_src)]

        for n, idx in enumerate(sel):
            sample = self.samples[idx]
            frame_window, center_idx = self._pick_frame_window(sample["frames"])
            center_time = (center_idx + 0.5) / self.fps

            if self.arch_frame == "clip":
                frames[n] = self._load_frames_clip(frame_window)
            else:
                frames[n] = self._load_frames(frame_window)
            audios[n] = self._load_audio(sample["audio"], center_time)
            infos[n] = [sample["audio"], sample["frame_dir"], str(len(sample["frames"]))]

        mag_mix, mags, phase_mix, audio_mix = self._mix_n_and_stft(audios)
        ret = {
            "mag_mix": mag_mix,
            "frames": frames,
            "mags": mags,
            "audio_mix": audio_mix,
            "audios": audios,
            "phase_mix": phase_mix,
            "infos": infos,
        }
        return ret


class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net = nets
        self.scale_factor = 0.15
        self.t_eps = 0.0
        self.sigma_min = 1e-4
        self.nfe_steps = 2
        self.loss_fn = lambda a, b: torch.mean(torch.abs(a - b))

    def euler_solver(self, x0, condition):
        self.net.eval()
        ts = torch.linspace(self.t_eps, 1, self.nfe_steps + 1)[1:]
        t = self.t_eps
        x = x0
        for tau in ts:
            dt = tau - t
            t_tensor = torch.ones(x.shape[0], device=x.device) * t
            dphi_dt = self.net(x, t_tensor * 1000, condition)
            x = x + dt * dphi_dt
            t = tau
        return x

    def sample(self, batch_data, args):
        mag_mix = batch_data["mag_mix"]
        mags = batch_data["mags"]
        frames = batch_data["frames"]
        mag_mix = mag_mix + 1e-10

        n_src = args.num_mix
        batch_size = mag_mix.size(0)
        time_steps = mag_mix.size(3)

        if args.log_freq:
            grid_warp = torch.from_numpy(warpgrid(batch_size, 256, time_steps, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix.to(args.device), grid_warp, align_corners=True)
            for n in range(n_src):
                mags[n] = F.grid_sample(mags[n].float().to(args.device), grid_warp, align_corners=True)

        log_mag_mix = torch.log1p(mag_mix) * self.scale_factor
        log_mag_mix.clamp_(0.0, 1.0)
        log_mag_mix = log_mag_mix.detach()

        feat_frames = [None for _ in range(n_src)]
        for n in range(n_src):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n].to(args.device), pool=False)

        x0 = torch.randn_like(log_mag_mix)
        x0 = self.t_eps * log_mag_mix + (1 - (1 - self.sigma_min) * self.t_eps) * x0

        pred_mags = [None for _ in range(n_src)]
        for n in range(n_src):
            pred = self.euler_solver(x0, condition=[log_mag_mix, feat_frames[n], log_mag_mix])
            pred = pred / self.scale_factor
            pred_mags[n] = torch.exp(pred.abs()) - 1

        return {"pred_mags": pred_mags, "mag_mix": mag_mix, "mags": mags}


def calc_metrics(batch_data, outputs, args):
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    mag_mix = batch_data["mag_mix"]
    phase_mix = batch_data["phase_mix"]
    audios = batch_data["audios"]
    pred_mags = outputs["pred_mags"].copy()

    n_src = args.num_mix
    batch_size = mag_mix.size(0)
    for n in range(n_src):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(batch_size, args.stft_frame // 2 + 1, mag_mix.size(3), warp=False)
            ).to(args.device)
            pred_mags[n] = F.grid_sample(pred_mags[n], grid_unwarp, align_corners=True)

    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(n_src):
        pred_mags[n] = pred_mags[n].detach().cpu().numpy()

    for j in range(batch_size):
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        preds_wav = [None for _ in range(n_src)]
        for n in range(n_src):
            pred_mag = pred_mags[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        length = preds_wav[0].shape[0]
        gts_wav = [None for _ in range(n_src)]
        valid = True
        for n in range(n_src):
            gts_wav[n] = audios[n][j, 0:length].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5

        if valid:
            sdr, sir, sar, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray(preds_wav), False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:length] for _ in range(n_src)]),
                False,
            )
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [
        sdr_mix_meter.average(),
        sdr_meter.average(),
        sir_meter.average(),
        sar_meter.average(),
    ]


def evaluate(net_wrapper, loader, args):
    print("Evaluating VGG directory dataset...")
    torch.set_grad_enabled(False)
    net_wrapper.eval()

    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    for _, batch_data in enumerate(loader):
        outputs = net_wrapper.module.sample(batch_data, args)
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

    print(
        "[Eval Summary] SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}".format(
            sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average(),
        )
    )


def parse_vgg_arguments():
    parser_wrapper = ArgParser()
    parser_wrapper.add_train_arguments()
    parser = parser_wrapper.parser
    parser.add_argument("--audio_dir", required=True, help="Root directory containing source audio files")
    parser.add_argument("--video_dir", required=True, help="Root directory containing extracted video frames")
    parser.add_argument(
        "--audio_exts",
        default=".wav,.flac,.mp3",
        help="Comma-separated list of allowed audio extensions",
    )
    args = parser.parse_args()
    parser_wrapper.print_arguments(args)
    return args


def main(args):
    if args.mode != "eval":
        raise ValueError("main_vgg.py supports eval mode only. Set --mode eval.")

    builder = models.ModelBuilder()
    net_frame = builder.build_visual(
        pool_type=args.img_pool,
        weights=args.weights_frame,
        arch_frame=args.arch_frame,
    )
    net_unet = builder.build_unet(weights=args.weights_unet)
    nets = (net_frame, net_unet)

    dataset_val = VGGDirMixDataset(args, split=args.split)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
    )

    net_wrapper = NetWrapper(nets)
    net_wrapper = torch.nn.DataParallel(net_wrapper)
    net_wrapper.to(args.device)

    evaluate(net_wrapper, loader_val, args)
    print("Evaluation Done!")


if __name__ == "__main__":
    args = parse_vgg_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    args.id += "-{}mix".format(args.num_mix)
    if args.log_freq:
        args.id += "-LogFreq"
    args.id += "-frames{}stride{}".format(args.num_frames, args.stride_frames)
    args.id += "-channels{}".format(args.num_channels)

    print("Model ID: {}".format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == "eval":
        args.weights_unet = os.path.join(args.ckpt, "unet_best.pth")
        args.weights_frame = os.path.join(args.ckpt, "frame_best.pth")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tic = time.perf_counter()
    main(args)
    print("Finished in {:.2f} seconds".format(time.perf_counter() - tic))