import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_downscale(
    n_in, n_out, kernel_size=4, normalization=nn.BatchNorm3d, activation=nn.ReLU
):
    block = nn.Sequential(
        nn.Conv3d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 2) // 2,
            bias=False,
        ),
        normalization(n_out),
        activation(inplace=True),
    )
    return block


class ResBlock(nn.Module):
    def __init__(
        self, n_out, kernel=3, normalization=nn.BatchNorm3d, activation=nn.ReLU
    ):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv3d(
                n_out, n_out, kernel, stride=1, padding=(kernel // 2), bias=False
            ),
            normalization(n_out),
            activation(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.Conv3d(
                n_out, n_out, kernel, stride=1, padding=(kernel // 2), bias=False
            ),
            normalization(n_out),
        )

        self.block2 = nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)

        x = self.block2(x + x0)
        return x


class Encode(nn.Module):
    def __init__(self):
        super().__init__()
        nf0 = 8 * 4
        nf1 = 16 * 4
        nf2 = 32 * 4
        self.block0 = nn.Sequential(nn.BatchNorm3d(1),)

        # output 16^3
        self.block1 = nn.Sequential(
            make_downscale(1, nf0, kernel_size=8), ResBlock(nf0), ResBlock(nf0),
        )

        # output 8^3
        self.block2 = nn.Sequential(
            make_downscale(nf0, nf1), ResBlock(nf1), ResBlock(nf1),
        )

        # output 4^3
        self.block3 = nn.Sequential(
            make_downscale(nf1, nf2), ResBlock(nf2), ResBlock(nf2),
        )

        self.proj = nn.Sequential(nn.Linear(4 * 4 * 4 * nf2 + 1, 128),)

    def forward(self, x):
        x_ = x.reshape(x.shape[0], -1)
        scan_scales = x_[range(x.shape[0]), x_.argmax(dim=1)].unsqueeze(1)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, scan_scales], dim=-1)
        x = nn.functional.normalize(self.proj(x))
        return x


# ########################## main model ##################################
# This is the CAD Encoder from Scan2CAD with n_channels * 4
class Model3d(nn.Module):
    def __init__(self):
        super(Model3d, self).__init__()
        self.encode = Encode()

    def forward(self, x):
        z = self.encode(x)
        return z


class ResNetBlock(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_size=3,
        stride=1,
        layer=nn.Conv3d,
        normalization=nn.BatchNorm3d,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.layer = layer
        self.normalization = normalization
        self.activation = activation

        # Full pre-activation block
        self.weight_block_0 = nn.Sequential(
            self.normalization(self.num_channels),
            self.activation(inplace=True),
            self.layer(
                self.num_channels,
                self.num_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            ),
        )

        self.weight_block_1 = nn.Sequential(
            self.normalization(self.num_channels),
            self.activation(inplace=True),
            self.layer(
                self.num_channels,
                self.num_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            ),
        )

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.weight_block_0(x)
        out = self.weight_block_1(out)
        out = identity + out
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, num_input_channels, num_features=None, verbose=False):
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.verbose = verbose
        self.num_features = [num_input_channels] + num_features
        self.network = nn.Sequential(
            # 32 x 32 x 32
            nn.Conv3d(
                self.num_features[0],
                self.num_features[1],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            # 32 x 32 x 32
            nn.Conv3d(
                self.num_features[1],
                self.num_features[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResNetBlock(self.num_features[1]),
            # 16 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.Conv3d(
                self.num_features[1],
                self.num_features[2],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResNetBlock(self.num_features[2]),
            # 8 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),
            nn.Conv3d(
                self.num_features[2],
                self.num_features[3],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResNetBlock(self.num_features[3]),
            # 4 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),
            nn.Conv3d(
                self.num_features[3],
                self.num_features[4],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResNetBlock(self.num_features[4]),
            # 2 x 2 x 2
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[4]),
            nn.Conv3d(
                self.num_features[4],
                self.num_features[5],
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(self.num_features[5]),
        )
        self.projector = nn.Linear(self.num_features[5], 128)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layers = list(self.network.children())
        for depth, layer in enumerate(layers):
            shape_before = x.data[0].size()
            x = layer(x)
            shape_after = x.data[0].size()

            if self.verbose is True:
                print(f"Layer {depth}: {shape_before} --> {shape_after}")
                self.verbose = False
        x = self.projector(x.reshape(x.shape[0], -1))
        return x


# PerturbedTopK module adapted from Cordonnier et al. [https://arxiv.org/abs/2104.03059].
class PerturbedTopK(nn.Module):
    def __init__(self, k, num_samples=1000, sigma=0.05):
        super(PerturbedTopK, self).__init__()
        self.k = k
        self.num_samples = num_samples
        self.sigma = sigma

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, num_samples=1000, sigma=0.05):
        b, d = x.shape

        if x.device == "cpu":
            noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        else:
            noise = torch.cuda.FloatTensor((b, num_samples, d))
            torch.normal(mean=0, std=1, size=(b, num_samples, d), out=noise)

        perturbed_x = x[:, None, :] + noise * sigma
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=True)
        indices = topk_results.indices  # b, nS, k
        perturbed_output = F.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        # Corrected gradient computation
        grad_expected = torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, ctx.noise)
        grad_expected /= (ctx.num_samples * ctx.sigma)
        grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)

        return (grad_input, ) + tuple([None] * 5)
