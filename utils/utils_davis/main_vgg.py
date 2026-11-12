# System libs
import os
import random
import sys

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
from imageio import imwrite as imsave

# submodule/DAVIS provides arguments/modules; its dataset/ dir provides the flat video_transforms module
DAVIS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'submodule', 'DAVIS'))
sys.path.insert(0, DAVIS_DIR)
sys.path.append(os.path.join(DAVIS_DIR, 'dataset'))

# Our libs
from arguments import ArgParser
from modules import models
from ave_vgg import VGGInferenceDataset
from utils_vgg import magnitude2heatmap, istft_reconstruction, warpgrid, makedirs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Network wrapper, defines the sampling pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net = nets
        self.scale_factor = 0.15
        self.t_eps = 0.
        self.sigma_min = 1e-4
        self.nfe_steps = 2

    def euler_solver(self, x0, condition):
        self.net.eval()
        ts = torch.linspace(self.t_eps, 1, self.nfe_steps+1)[1:]
        t = self.t_eps 
        x = x0

        # mask = (condition[0] > 0.002).float()

        for tau in ts:
            dt = tau - t
            t = torch.ones(x.shape[0], device=x.device) * t
            dphi_dt = self.net(x, t*1000, condition)
            x = x + dt * dphi_dt
            # # silence_mask_guided_sampling
            # x = mask * (x + dt * dphi_dt) + (1-mask) * (tau * condition[0] + (1 - (1 - self.sigma_min) * tau) * x0)
            t = tau
        return x

    def sample_single(self, batch_data, args):
        mag_mix = batch_data['mag_mix'].to(args.device) + 1e-10
        B = mag_mix.size(0)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, 256, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # LOG magnitude
        log_mag_mix = (torch.log1p(mag_mix) * self.scale_factor).clamp_(0., 1.).detach()

        # Frame feature (condition)
        feat_frames = self.net_frame.forward_multiframe(
            batch_data['frames'].to(args.device), pool=False)

        # euler solver sampling
        x0 = torch.randn_like(log_mag_mix)
        x0 = self.t_eps * log_mag_mix + (1 - (1 - self.sigma_min) * self.t_eps) * x0
        pred = self.euler_solver(x0, condition=[log_mag_mix, feat_frames, log_mag_mix])

        pred_mag = torch.exp((pred / self.scale_factor).abs()) - 1
        return {'pred_mag': pred_mag, 'mag_mix': mag_mix}


# Write separated wavs and visualizations; returns (written, skipped, num_saved_vis)
def save_outputs(batch_data, outputs, args, num_saved_vis):
    mag_mix = batch_data['mag_mix']
    B = mag_mix.size(0)

    pred_mag = outputs['pred_mag']
    if args.log_freq:
        grid_unwarp = torch.from_numpy(
            warpgrid(B, args.stft_frame//2+1, mag_mix.size(3), warp=False)).to(args.device)
        pred_mag = F.grid_sample(pred_mag, grid_unwarp, align_corners=True)

    pred_mag = pred_mag.detach().cpu().numpy()
    mag_mix = mag_mix.numpy()
    phase_mix = batch_data['phase_mix'].numpy()

    written, skipped = 0, 0
    for j in range(B):
        name = batch_data['name'][j]
        if int(batch_data['valid'][j]) == 0:
            print('Skipping invalid clip: {}'.format(name))
            skipped += 1
            continue

        pred_wav = istft_reconstruction(pred_mag[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        wavfile.write(os.path.join(args.out_dir, 'separated', name + '.wav'),
                      args.audRate, pred_wav.astype(np.float32))
        written += 1

        if num_saved_vis < args.num_vis:
            vis_dir = os.path.join(args.out_dir, 'visualization', name)
            makedirs(vis_dir)
            mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
            wavfile.write(os.path.join(vis_dir, 'mix.wav'),
                          args.audRate, mix_wav.astype(np.float32))
            imsave(os.path.join(vis_dir, 'mix.jpg'),
                   magnitude2heatmap(mag_mix[j, 0])[::-1, :, :])
            imsave(os.path.join(vis_dir, 'pred.jpg'),
                   magnitude2heatmap(pred_mag[j, 0])[::-1, :, :])
            num_saved_vis += 1

    return written, skipped, num_saved_vis


def main(args):
    # Network Builders
    builder = models.ModelBuilder()
    net_frame = builder.build_visual(
        pool_type=args.img_pool,
        weights=args.weights_frame,
        arch_frame=args.arch_frame)
    net_unet = builder.build_unet(weights=args.weights_unet)

    netWrapper = NetWrapper((net_frame, net_unet))
    netWrapper.to(args.device)
    netWrapper.eval()

    # Dataset and Loader
    dataset = VGGInferenceDataset(
        args.list_val, args, max_sample=args.num_val, split='val')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)

    total_written, total_skipped, num_saved_vis = 0, 0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            outputs = netWrapper.sample_single(batch_data, args)
            written, skipped, num_saved_vis = save_outputs(
                batch_data, outputs, args, num_saved_vis)
            total_written += written
            total_skipped += skipped

            if i % args.disp_iter == 0:
                print('[{}/{}] written: {}, skipped: {}'.format(
                    i + 1, len(loader), total_written, total_skipped))

    print('Done. Written: {}, skipped: {}, output: {}'.format(
        total_written, total_skipped, args.out_dir))


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    parser.add_train_arguments()
    parser.parser.add_argument('--audio_root', required=True,
                               help='folder holding the wav files listed in --list_val')
    parser.parser.add_argument('--frame_root', required=True,
                               help='folder holding one frame folder per clip')
    parser.parser.add_argument('--out_dir', required=True,
                               help='output folder for separated wavs and visualizations')
    args = parser.parser.parse_args()
    parser.print_arguments(args)

    if not args.weights_frame or not args.weights_unet:
        sys.exit('--weights_frame and --weights_unet are required.')

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    makedirs(os.path.join(args.out_dir, 'separated'), remove=False)
    makedirs(os.path.join(args.out_dir, 'visualization'), remove=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
