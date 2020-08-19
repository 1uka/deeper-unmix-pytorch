import torch
import torch.nn as nn
import torch.nn.functional as F


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)


class EncoderBlock2d(nn.Module):
    """A simple encoder block for a frequency filter convolutional network using 2D convolution.

    Meant to be applied to time-frequency spectrograms of audio samples.
    Uses one convolution layer that reduces the dimensionality, and a subsequent
    rewrite convolution (or the actual filter) which will double the output channels which is
    needed because of the GLU activation function.
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """Initialize the EncoderBlock2d module

        Parameters
        ----------
        in_channels : int
            input channels dimensionality for Conv2d
        out_channels : int
            output channels dimensionality for Conv2d
        """

        super(EncoderBlock2d, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        output = self.encode(x)

        return output


class SFFilter(nn.Module):
    """A frequency filter module for audio spectrograms, using multiple EncoderBlock2d.

    The goal is to bring down the input spectrogram to a latent, 2D filter that can
    be applied over a different spectrogram.
    """

    def __init__(self, channels, bins, *args, **kwargs):
        """Initialize the SFF

        This module expects input with shape (batch_size, channels, bins, frames),
        and gives (batch_size, channels, bins) shaped output.

        Parameters
        ----------
        channels : int
            number of audio channels, input to the first block (1 for mono, 2 for stereo)
        bins : int
            number of frequency bins, also its the output dimension
        """
        super(SFFilter, self).__init__()

        self.channels = channels
        self.bins = bins

        blocks = []
        in_channels = channels
        out_channels = 64
        layers = 4
        for _ in range(layers):
            blocks += [EncoderBlock2d(in_channels, out_channels,
                                      *args, **kwargs), nn.MaxPool2d(2, 2)]
            in_channels = out_channels
            out_channels *= 2

        blocks += [EncoderBlock2d(in_channels, bins,
                                  kernel_size=8, stride=3, padding=8)]

        self.network = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(3, 3)

    def forward(self, x):
        x = self.pool(self.network(x))

        # we want last dimension to be 1 always, and penultimate to be
        # the number of audio channels, which is anyways accomplished with
        # the convolution
        x = F.relu(torch.mean(x, -1))
        x = x.reshape(-1, self.channels, self.bins)

        return x


if __name__ == "__main__":
    # test if the architecture works

    s = torch.randn(1, 2, 6 * 44100)

    stft = STFT(n_fft=4096, n_hop=1024)
    spec = Spectrogram(power=1, mono=False)

    transform = nn.Sequential(stft, spec)

    kernel_size = 4
    padding = kernel_size // 2
    ff = SFFilter(2, 2049, kernel_size=kernel_size,
                  stride=2, padding=padding)

    x = transform(s).permute(1, 2, 3, 0)
    print("Shape of spectrogram of x: ", x.shape)

    x = ff(x)
    print("Shape after SFF", x.shape)
