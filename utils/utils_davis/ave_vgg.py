import os
import torch
from base_vgg import BaseDataset


class VGGInferenceDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(VGGInferenceDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.audLen = opt.audLen
        self.model_type = opt.arch_frame
        self.audio_root = opt.audio_root
        self.frame_root = opt.frame_root

    def __getitem__(self, index):
        path_audio, path_frame, count_frames = self.list_sample[index][:3]
        count_frames = int(count_frames)
        name = os.path.splitext(os.path.basename(path_audio))[0]

        fps = count_frames // 10
        self.stride_frames = fps
        center_frame = 6

        path_frames = []
        for i in range(self.num_frames):
            fid = center_frame + (i - self.num_frames // 2) * fps
            # clips shorter than 11 frames repeat their last frame
            fid = min(max(fid, 1), count_frames)
            path_frames.append(os.path.join(
                self.frame_root, path_frame, '{:06d}.jpg'.format(fid)))
        path_audio = os.path.join(self.audio_root, path_audio)

        try:
            if self.model_type != 'clip':
                frames = self._load_frames(path_frames)
            else:
                frames = self._load_frames_clip(path_frames)

            # the wav already is the mixture
            audio = self._load_audio(path_audio, center_timestamp=5)
            amp, phase = self._stft(audio)
            mag_mix = amp.unsqueeze(0)
            phase_mix = phase.unsqueeze(0)
            audio_mix = audio
            valid = 1
        except Exception as e:
            print('Failed loading frame/audio {}: {}'.format(name, e))
            # real STFT width; self.WS is one short
            width = self.audLen // self.stft_hop + 1
            mag_mix = torch.zeros(1, self.HS, width)
            phase_mix = torch.zeros(1, self.HS, width)
            frames = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audio_mix = torch.zeros(self.audLen)
            valid = 0

        return {'mag_mix': mag_mix, 'phase_mix': phase_mix, 'frames': frames,
                'audio_mix': audio_mix, 'name': name, 'valid': valid}
