import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderBlock2d, DecoderBlock2d, VQEmbeddingEMA, STFT, Spectrogram, NoOp
from utils import center_trim


class VQVadass(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        num_embeddings=512,
        nb_channels=2,
        sample_rate=44100,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        power=1,
        kernel_size=3,
        stride=2,
        dilation=1,
        context=3,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(VQVadass, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.num_embeddings = num_embeddings
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        # VQ-VAE model architecture
        # input shape: (B, C, N, F)
        padding = (kernel_size - stride) // 2 * dilation
        self.enc1 = EncoderBlock2d(nb_channels, self.nb_bins // 4, kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc2 = EncoderBlock2d(self.nb_bins // 4, self.nb_bins // 2, kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc3 = EncoderBlock2d(self.nb_bins // 2, self.nb_bins, kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)

        self.vq1 = VQEmbeddingEMA(num_embeddings, self.nb_bins)

        self.dec1 = DecoderBlock2d(self.nb_bins, self.nb_bins // 2, context)
        self.dec2 = DecoderBlock2d(
            self.nb_bins // 2, self.nb_bins // 4, context)
        self.dec3 = DecoderBlock2d(self.nb_bins // 4, nb_channels, context)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)

        self.output_scale = nn.Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = nn.Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # crop (if max_bins is set on self.nb_bins)
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # output from transform has shape (F, B, C, N)
        # but we want to convolute on B x (C x F x N), so we need to reshape
        x = x.view(nb_samples, nb_channels, nb_frames, self.nb_bins)

        # pass through encoder and save skip connections
        skip_conns = []
        x = self.enc1(x)
        skip_conns.append(x)
        x = self.enc2(x)
        skip_conns.append(x)
        x = self.enc3(x)
        skip_conns.append(x)
        folded_shape = x.shape[-2:]
        x = F.unfold(x, kernel_size=1)

        # apply VQ embedding
        loss, x, _, _ = self.vq1(x)

        x = F.fold(x, folded_shape, kernel_size=1)

        # decode and apply skip connections
        x = self.dec1(x, skip=skip_conns.pop())
        x = self.dec2(x, skip=skip_conns.pop())
        x = self.dec3(x, skip=skip_conns.pop())

        # pad what was lost in strided convolution (just a few dimensions hopefully)
        pad_N = abs(x.size(-1) - self.nb_output_bins)
        pad_F = abs(x.size(-2) - nb_frames)
        x = F.pad(x, (0, pad_N, 0, pad_F), "constant", 0)

        # reshape OpenUnmix to output dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x)

        return x


if __name__ == "__main__":
    T = 44100 * 10
    x = torch.rand(1, 2, T)

    net = VQVadass()
    y = net(x)
