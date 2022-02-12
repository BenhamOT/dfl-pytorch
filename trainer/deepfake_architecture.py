import torch
from torch import nn
from torch.nn import functional as F


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        # print("Downscale in_ch is {}".format(in_ch))
        # print("Downscale out_ch is {}".format(out_ch))
        # print("Downscale kernal_size is {}".format(kernel_size))
        super(Downscale, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=kernel_size, stride=2, padding=2)

    def forward(self, x):
        # print("Downscale input shape is {}".format(x.shape))
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        # print("Downscale output shape is {}".format(x.shape))
        return x

    def get_out_ch(self):
        return self.out_ch


class DownscaleBlock(nn.Module):

    def __init__(self, in_ch, ch, n_downscales, kernel_size):
        # print("DownscaleBlock in_ch is {}".format(in_ch))
        # print("DownscaleBlock ch is {}".format(ch))
        # print("Downscale n_downscales is {}".format(n_downscales))
        super(DownscaleBlock, self).__init__()
        self.downs = []

        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * (min(2 ** i, 8))
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size))
            last_ch = self.downs[-1].get_out_ch()

    def forward(self, inp):
        x = inp
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(Upscale, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding="same")

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pixel_shuffle(x, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp):
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, e_ch):
        # print("Encoder in_ch is {}".format(in_ch))
        # print("Encoder e_ch is {}".format(in_ch))
        super(Encoder, self).__init__()
        self.down = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5)
        self.flatten = nn.Flatten()

    def forward(self, inp):
        x = self.down(inp)
        print("pre flatten shape is {}".format(x.shape))
        return self.flatten(x)


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, resolution):
        super(Inter, self).__init__()
        self.in_ch = in_ch
        self.ae_ch = ae_ch
        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = resolution // 16
        print("Inter in_ch shape is {}".format(self.in_ch))
        print("Inter ae_ch shape is {}".format(self.ae_ch))
        print("Inter ae_ch_out shape is {}".format(self.ae_out_ch))

        self.dense1 = nn.Linear(self.in_ch, self.ae_ch)
        self.dense2 = nn.Linear(ae_ch, self.lowest_dense_res * self.lowest_dense_res * self.ae_out_ch)
        self.upscale1 = Upscale(self.ae_out_ch, self.ae_out_ch)  # does the kernel size need to be included here?

    def forward(self, inp):
        x = inp
        print("The inter input shape is {}".format(x.shape))
        x = self.dense1(x)
        print("The shape after dense1 is {}".format(x.shape))
        x = self.dense2(x)
        print("The shape before the view is {}".format(x.shape))
        x = x.view(x.shape[0], self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)
        # x = nn.reshape_4D(x, self.lowest_dense_res, self.lowest_dense_res, self.ae_out_ch)
        print("shape before upscale is {}".format(x.shape))
        x = self.upscale1(x)
        return x

    def get_code_res(self):
        return self.lowest_dense_res

    def get_out_ch(self):
        return self.ae_out_ch


class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch):
        super(Decoder, self).__init__()
        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 2, kernel_size=3)
        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1, padding="same")

        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3)
        self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1, padding="same")

    def forward(self, inp):
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


