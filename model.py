import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderBlock2d, DecoderBlock2d, VQEmbeddingEMA, STFT, Spectrogram, NoOp
from utils import center_trim


class Vaess(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        latent_dim=128,
        nb_channels=2,
        sample_rate=44100,
        seq_dur=6.0,
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

        self.hidden_dims = [32, 64, 128, 256]

        # VAE U-net model architecture
        # input shape: (B, C, N, F)
        padding = (kernel_size - stride) // 2 * dilation
        self.enc1 = EncoderBlock2d(nb_channels, self.hidden_dims[0], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc2 = EncoderBlock2d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc3 = EncoderBlock2d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc4 = EncoderBlock2d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)

        encoded_size = self._calculate_flat_dim(
            self.nb_bins, kernel_size, padding, stride, dilation, 4)
        n_frames = int(((sample_rate * seq_dur) - n_fft) // n_hop + 1)
        encoded_size *= self._calculate_flat_dim(
            n_frames, kernel_size, padding, stride, dilation, 4)
        encoded_size *= self.hidden_dims[-1]

        self.fc_mu = nn.Linear(encoded_size, latent_dim)
        self.fc_var = nn.Linear(encoded_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, encoded_size)

        self.dec1 = DecoderBlock2d(
            self.hidden_dims[3], self.hidden_dims[2], context)
        self.dec2 = DecoderBlock2d(
            self.hidden_dims[2], self.hidden_dims[1], context)
        self.dec3 = DecoderBlock2d(
            self.hidden_dims[1], self.hidden_dims[0], context)
        self.dec4 = DecoderBlock2d(self.hidden_dims[0], nb_channels, context)

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

        return w

    def encode(self, x):
        skip_conns = []
        x = self.enc1(x)
        skip_conns.append(x)
        x = self.enc2(x)
        skip_conns.append(x)
        x = self.enc3(x)
        skip_conns.append(x)
        x = self.enc4(x)
        skip_conns.append(x)

        encoded_shape = x.data.shape
        x = torch.flatten(x, start_dim=1)

        mu, logvar = self.fc_mu(x), self.fc_var(x)

        return mu, logvar, skip_conns, encoded_shape

    def decode(self, x, skip_conns, encoded_shape):
        x = self.fc_dec(x)
        x = x.view(encoded_shape)

        x = self.dec1(x, skip=skip_conns.pop().detach())
        x = self.dec2(x, skip=skip_conns.pop().detach())
        x = self.dec3(x, skip=skip_conns.pop().detach())
        x = self.dec4(x, skip=skip_conns.pop().detach())

        return x

    def reparametarize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x):
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        x = x[..., :self.nb_bins]

        x += self.input_mean
        x *= self.input_scale

        x = x.reshape(nb_samples, nb_channels, nb_frames, self.nb_bins)

        mu, logvar, skip_conns, encoded_shape = self.encode(x)
        z = self.reparametarize(mu, logvar)
        recon_x = self.decode(z, skip_conns, encoded_shape)

        # pad what was lost in strided convolution (just a few dimensions hopefully)
        pad_N = abs(recon_x.size(-1) - self.nb_output_bins)
        pad_F = abs(recon_x.size(-2) - nb_frames)
        recon_x = F.pad(
            recon_x, (0, pad_N, 0, pad_F), "constant", 0)

        recon_x = recon_x.reshape(
            nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        recon_x *= self.output_scale
        recon_x += self.output_mean

        recon_x = F.relu(recon_x) * mix

        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def generate(self, x):
        return self.forward(x)[0]
