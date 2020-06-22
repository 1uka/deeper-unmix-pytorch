import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderBlock2d, DecoderBlock2d, STFT, Spectrogram, NoOp
from utils import center_trim


class Vaess(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        latent_dim=512,
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

        super(Vaess, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        # actual formula is ((sample_rate * seq_dur) // n_hop) + 1
        # but we cannot know seq_dur at runtime so we use this value
        # to set the reshaping encoded size for embedding layer
        self.nb_frames = int((sample_rate - n_fft) // n_hop) + 1

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.latent_dim = latent_dim
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.hidden_dims = [64, 128, 256]

        # VAE U-net model architecture
        # input shape: (B, C, N, F)
        padding = (kernel_size - stride) // 2 * dilation
        self.enc1 = EncoderBlock2d(nb_channels, self.hidden_dims[0], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc2 = EncoderBlock2d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc3 = EncoderBlock2d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)

        self.fc1 = nn.Linear(
            in_features=self.hidden_dims[-1], out_features=latent_dim, bias=False)

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim // 2,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.2,
        )

        self.fc2 = nn.Linear(in_features=latent_dim * 2,
                             out_features=self.hidden_dims[-1], bias=False)

        self.dec1 = DecoderBlock2d(
            self.hidden_dims[2], self.hidden_dims[1], context)
        self.dec2 = DecoderBlock2d(
            self.hidden_dims[1], self.hidden_dims[0], context)
        self.dec3 = DecoderBlock2d(
            self.hidden_dims[0], nb_channels, context)

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

    def _calculate_flat_dim(self, W, F, P, S, D, L):
        w = W
        for _ in range(L):
            w = (w + 2*P - D*(F - 1) - 1) // S

        return abs(w)

    def encode(self, x):
        skip_conns = []
        x = self.enc1(x)
        skip_conns.append(x)
        x = self.enc2(x)
        skip_conns.append(x)
        x = self.enc3(x)
        skip_conns.append(x)

        return x, skip_conns

    def decode(self, x, skip_conns=None):
        if skip_conns is None:
            x = self.dec1(x)
            x = self.dec2(x)
            x = self.dec3(x)
        else:
            x = self.dec1(x, skip=skip_conns.pop().detach())
            x = self.dec2(x, skip=skip_conns.pop().detach())
            x = self.dec3(x, skip=skip_conns.pop().detach())

        return x

    def forward(self, x):
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        x = x[..., :self.nb_bins]

        x += self.input_mean
        x *= self.input_scale

        x = x.reshape(nb_samples, nb_channels, self.nb_bins, nb_frames)

        x, skip_conns = self.encode(x)
        encoded_shape = x.data.shape

        # rotate and embed with fc1
        x = self.fc1(x.view(nb_samples, -1, self.hidden_dims[-1]))
        x = torch.tanh(x)

        # BLSTM step
        lstm_out = self.lstm(x)

        # add LSTM skip connection, fc2, rotate back and reconstruct
        x = torch.cat([x, lstm_out[0]], -1)
        x = F.relu(self.fc2(x))
        x = x.view(encoded_shape)
        recon_x = self.decode(x, skip_conns)

        # pad what was lost in strided convolution (just a few dimensions hopefully)
        pad_F = abs(recon_x.size(-1) - nb_frames)
        pad_N = abs(recon_x.size(-2) - self.nb_output_bins)

        recon_x = F.pad(
            recon_x, (0, pad_F, 0, pad_N), "constant", 0)

        recon_x = recon_x.reshape(
            nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        recon_x *= self.output_scale
        recon_x += self.output_mean

        recon_x = F.relu(recon_x) * mix

        return recon_x

    def loss_function(self, recon_x, x):
        MSE = F.mse_loss(recon_x, x)

        return MSE
