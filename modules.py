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
            (B, C, N, F, 2)
        Output: Power/Mag Spectrogram
            (F, B, C, N)
        """

        # reshape to (B, C, F, N, 2)
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude -> (B, C, F, N)
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        return stft_f.permute(2, 0, 1, 3)


class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(EncoderBlock2d, self).__init__()

        self.encode_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        x = self.encode_1(x)

        return x


class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, context):
        super(DecoderBlock2d, self).__init__()

        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=context,
                      stride=1, padding=context // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            diffX = skip.size(-1) - x.size(-1)
            diffY = skip.size(-2) - x.size(-2)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

            x = torch.tanh(x) * torch.sigmoid(skip)

        x = self.ct1(x)
        x = self.conv1(x)

        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(self, N, E, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super(VQEmbeddingEMA, self).__init__()

        self.N = N
        self.E = E

        self.embedding = nn.Embedding(N, E)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(N))
        self.ema_w = nn.Parameter(torch.Tensor(N, E))
        self.ema_w.data.normal_()

        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

    def forward(self, inputs):
        # convert inputs from (B, C, T) -> (B, T, C)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.E)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight **
                                                                               2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.N, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.N * self.eps) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(
                self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from (B, T, C) -> (B, C, T)
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings
