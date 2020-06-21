import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderBlock1d, DecoderBlock1d
from utils import center_trim


class Vaess(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        nb_channels=2,
        sample_rate=44100,
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

        self.latent_dim = latent_dim
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        self.hidden_dims = [32, 64, 128, 256]

        # VAE U-net model architecture
        # input shape: (B, C, N, F)
        padding = (kernel_size - stride) // 2 * dilation
        self.enc1 = EncoderBlock1d(nb_channels, self.hidden_dims[0], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc2 = EncoderBlock1d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc3 = EncoderBlock1d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)
        self.enc4 = EncoderBlock1d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=kernel_size,
                                   stride=stride, dilation=dilation, padding=padding)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.hidden_dims[-1])

        self.dec1 = DecoderBlock1d(
            self.hidden_dims[3], self.hidden_dims[2], context)
        self.dec2 = DecoderBlock1d(
            self.hidden_dims[2], self.hidden_dims[1], context)
        self.dec3 = DecoderBlock1d(
            self.hidden_dims[1], self.hidden_dims[0], context)
        self.dec4 = DecoderBlock1d(self.hidden_dims[0], nb_channels, context)

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
        x = self.enc4(x)
        skip_conns.append(x)

        # rotate so H is self.hidden_dims[-1]
        x = x.permute(0, 2, 1)

        mu, logvar = self.fc_mu(x), self.fc_var(x)

        return mu, logvar, skip_conns

    def decode(self, x, skip_conns=None):
        x = self.fc_dec(x)

        # rotate back so W is self.hidden_dims[-1] channels
        x = x.permute(0, 2, 1)

        if skip_conns is None:
            x = self.dec1(x)
            x = self.dec2(x)
            x = self.dec3(x)
            x = self.dec4(x)
        else:
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
        mu, logvar, skip_conns = self.encode(x)
        z = self.reparametarize(mu, logvar)
        recon_x = self.decode(z, skip_conns)

        # pad what was lost in strided convolution (just a few dimensions hopefully)
        pad_N = abs(recon_x.size(-1) - x.size(-1))
        recon_x = F.pad(recon_x, (0, pad_N), "constant", 0)

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

    def sample(self, num_samples, device):
        x = torch.randn(num_samples, self.latent_dim).to(device).detach()
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        z = self.reparametarize(mu, logvar)

        z = z.to(device).detach()
        output = self.decode(z, None)

        return output
