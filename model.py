from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
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


class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.conv1 = nn.Conv2d(nb_channels, 32, 5, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.bn1 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=True,
            dropout=0.2,
        )

        self.fc2 = nn.Linear(hidden_size, 128)

        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.ct1 = nn.ConvTranspose2d(128, 64, 5, 2, bias=False)
        self.ct2 = nn.ConvTranspose2d(64, 32, 5, 2, bias=False)
        self.ct3 = nn.ConvTranspose2d(32, nb_channels, 5, 2, bias=False)

        self.bn2 = nn.BatchNorm2d(nb_channels)

        # in case stft is cropped, dont pad with 0's; instead "learn the padding"
        self.fc3 = nn.Linear(self.nb_bins * nb_channels,
                             self.nb_output_bins * nb_channels)
        self.bn3(self.nb_output_bins * nb_channels)

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

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(nb_samples, nb_channels, self.nb_bins, nb_frames)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        unpooled_size = x.size()
        x, p = self.pool(self.bn1(x))
        encoded_shape = x.data.shape

        # rotate and embed
        x = torch.tanh(self.fc1(x.view(nb_samples, -1, 128)))

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)[0]

        # lstm skip connection
        x = x * torch.sigmoid(lstm_out)
        x = self.fc2(x)

        x = x.view(encoded_shape)

        x = self.unpool(x, p, output_size=unpooled_size)
        x = F.relu(self.ct1(x))
        x = F.relu(self.ct2(x))
        x = F.relu(self.ct3(x))
        x = self.bn2(x)

        # pad frames and cropped bins
        pad_F = abs(x.size(-1) - nb_frames)
        pad_N = abs(x.size(-2) - self.nb_bins)
        x = F.pad(x, (0, pad_F, 0, pad_N), "constant", 0)

        x = self.fc3(x.view(nb_frames, nb_samples, nb_channels * self.nb_bins))
        x = self.bn3(x)

        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x
