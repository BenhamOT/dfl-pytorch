import torch
from torch import nn
from torch.nn import functional as F


class Downscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=5) -> None:
        """
        Downscale component.

        Args:
            in_ch (int): The number of input channels e.g. 3 for a standard RGB image
            out_ch (int): The number of output channels
            kernel_size (int): The kernel size used during the convolution operation
        """
        super(Downscale, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(
            self.in_ch, self.out_ch, kernel_size=kernel_size, stride=2, padding=2
        )
        self.batch_norm = nn.BatchNorm2d(self.out_ch)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)
        return x

    def get_out_ch(self) -> int:
        return self.out_ch


class DownscaleBlock(nn.Module):
    def __init__(
        self, in_ch: int, ch: int, n_downscales: int, kernel_size: int
    ) -> None:
        """
        The Downscale block. The encoder part of the model. This constructs a sequence of Downscale components.

        Args:
            in_ch (int): The initial number of input channels for the first Downscale block
            ch (int): Used to calculate the number output channels
            n_downscales (int): Number of Downscale blocks
            kernel_size (int): The kernel size used during the convolution operation
        """
        super(DownscaleBlock, self).__init__()
        self.downs = nn.ModuleList()

        self.last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * (min(2**i, 8))
            self.downs.append(Downscale(self.last_ch, cur_ch, kernel_size=kernel_size))
            self.last_ch = self.downs[-1].get_out_ch()

    def get_out_ch(self) -> int:
        return self.last_ch

    def forward(self, inp: torch.tensor) -> torch.tensor:
        x = inp
        for down in self.downs:
            x = down(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int, e_ch: int) -> None:
        """
        The encoder component.

        Args:
            in_ch (int): The input channels fed to the DownscaleBlock
            e_ch (int): Used to calculate the inital number of output channels.
                This will control how the output channels in each of the Downscale blocks changes
        """
        super(Encoder, self).__init__()
        self.down = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5)
        self.flatten = nn.Flatten()

    def get_output_length(self, input_resolution: int) -> int:
        downscale_block_output_channels = self.down.get_out_ch()
        output_length = (
            downscale_block_output_channels * (input_resolution / 2**4) ** 2
        )
        return int(output_length)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        x = self.down(inp)
        return self.flatten(x)


class Upscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        """
        Deconvolution block used for upscaling the input tensor

        Args:
            in_ch (int): The number of input channels
            out_ch (int): The number of output channels
        """
        super(Upscale, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.batch_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.deconv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)
        return x


class Inter(nn.Module):
    def __init__(
        self, in_ch: int, ae_ch: int, ae_out_ch: int, lowest_dense_res: int
    ) -> None:
        """
        The intermediate step in the encoder decoder network. This takes in the flattened output from the encoder.
        Forward expects an input of shape [batch_size, in_ch]

        Args:
            in_ch (int): The length of the input vector
            ae_ch (int): The length of the output vector from the first dense layer
            ae_out_ch (int): The number of output channels in the upscaled tensor
            lowest_dense_res (int): The height and width of the tensor being passed into the Upscale block
        """
        super(Inter, self).__init__()
        self.in_ch = in_ch
        self.ae_ch = ae_ch
        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res

        self.dense1 = nn.Linear(self.in_ch, self.ae_ch)
        self.dense2 = nn.Linear(
            ae_ch, self.lowest_dense_res * self.lowest_dense_res * self.ae_out_ch
        )
        self.upscale1 = Upscale(self.ae_out_ch, self.ae_out_ch, kernel_size=3)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        x = inp
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(
            x.shape[0], self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res
        )
        x = self.upscale1(x)
        return x

    def get_code_res(self) -> int:
        return self.lowest_dense_res

    def get_out_ch(self) -> int:
        return self.ae_out_ch


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        """
        Residule convolutional block

        Args:
            ch (int): The number of input and output channels
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp: torch.tensor) -> torch.tensor:
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Decoder(nn.Module):
    def __init__(
        self, in_ch: int, d_ch: int, d_mask_ch: int, image_output_channels: int = 3
    ) -> None:
        """
        The docoder block

        Args:
            in_ch (int): The number of input channels
            d_ch (int): Parameter to control the number of output channels in each of the Upscale blocks
            d_mask_ch (int): Parameter to control the number of output channels in the mask Upscale block
            image_output_channels (int): The number of channels in the output image
        """
        super(Decoder, self).__init__()
        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 2, kernel_size=3)
        self.out_conv = nn.Conv2d(
            d_ch * 2, image_output_channels, kernel_size=1, padding="same"
        )

        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3)
        self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1, padding="same")

    def forward(self, inp: torch.tensor) -> torch.tensor:
        z = inp
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = torch.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = torch.sigmoid(self.out_convm(m))
        return x, m
