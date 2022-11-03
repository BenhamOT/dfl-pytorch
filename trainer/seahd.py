import torch
import torch.optim as optim
import torchshow as ts

from trainer.deepfake_architecture import Encoder, Decoder, Inter
from trainer.data_loader import CustomDataLoader
from trainer.utils import gaussian_blur, dssim
from params import Params


class SAEHDModel:
    def __init__(self):
        self.resolution = Params.resolution
        self.e_dims = Params.e_dims
        self.ae_dims = Params.ae_dims
        self.d_dims = Params.d_dims
        self.d_mask_dims = Params.d_mask_dims
        self.masked_training = Params.masked_training
        # self.learn_mask = Params.learn_mask
        self.eyes_priority = Params.eyes_priority
        # self.lr_dropout =
        # self.random_warp =
        # self.target_iterations =
        # self.random_flip =
        # self.batch_size =
        # self.pretrain =
        # self.uniform_yaw =
        # self.ct_mode =
        # self.clip_gradients =
        self.is_training = Params.is_training
        self.epochs = Params.epochs
        self.input_ch = Params.image_input_channels

        self.encoder = Encoder(in_ch=self.input_ch, e_ch=self.e_dims)
        encoder_out_ch = self.encoder.get_output_length(input_resolution=self.resolution)

        self.inter_AB = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            resolution=self.resolution
        )
        self.inter_B = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            resolution=self.resolution
        )

        inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        inters_out_ch = inter_AB_out_ch + inter_B_out_ch
        self.decoder = Decoder(in_ch=inters_out_ch, d_ch=self.d_dims, d_mask_ch=self.d_mask_dims)

        # self.encoder_optimiser = optim.RMSprop(self.encoder.parameters())
        # self.inter_B_optimiser = optim.RMSprop(self.inter_B.parameters())
        # self.inter_AB_optimiser = optim.RMSprop(self.inter_AB.parameters())
        # self.decoder_optimiser = optim.RMSprop(self.decoder.parameters())
        self.encoder_optimiser = optim.Adam(self.encoder.parameters())
        self.inter_B_optimiser = optim.Adam(self.inter_B.parameters())
        self.inter_AB_optimiser = optim.Adam(self.inter_AB.parameters())
        self.decoder_optimiser = optim.Adam(self.decoder.parameters())

    def run(self, src_path: str, dst_path: str):

        for i in range(self.epochs):
            print("epoch {}".format(i))

            src_average_losses = []
            dst_losses = []
            G_loss_gvs = []

            # TODO add in tqdm here when finsihed debugging
            for sample in CustomDataLoader(src_path=src_path, dst_path=dst_path).run():
                src_loss, dst_loss, combined_loss = self.train(sample=sample)
                src_average_losses.append(src_loss.mean())

                print("src loss is {}".format(src_loss))
                print("dst loss is {}".format(dst_loss))
                print("combined loss is {}".format(combined_loss))
            # sns.lineplot(x=src_average_losses, y=[i for i in range(len(src_average_losses))])
            # plt.show()

    def compute_loss(self, target, prediction, target_mask, predicted_mask, target_eyes_mask):
        if self.resolution < 256:
            dssim_result = 10 * dssim(
                img1=target,
                img2=prediction,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            loss = torch.mean(dssim_result, dim=1)
        else:
            dssim_result = 5 * dssim(
                img1=target,
                img2=prediction,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            loss = torch.mean(dssim_result, dim=1)
            dssim_result = 5 * dssim(
                img1=target,
                img2=prediction,
                max_val=1.0,
                filter_size=int(self.resolution / 23.2)
            )
            loss += torch.mean(dssim_result, dim=1)

        loss += torch.mean(10 * torch.square(target - prediction),
                                   dim=(1, 2, 3))

        if self.eyes_priority:
            loss += torch.mean(
                input=300 * torch.abs(
                    target * target_eyes_mask - prediction * target_eyes_mask),
                dim=(1, 2, 3)
            )

        loss += torch.mean(10 * torch.square(target_mask - predicted_mask), dim=(1, 2, 3))
        return loss

    def train(self, sample):

        self.encoder.zero_grad()
        self.inter_B.zero_grad()
        self.inter_AB.zero_grad()
        self.decoder.zero_grad()

        src_encoder_output = self.encoder(sample["warped_src"])
        src_inter_AB_output = self.inter_AB(src_encoder_output)
        src_inter_AB_concat = torch.cat((src_inter_AB_output, src_inter_AB_output), dim=1)

        dst_encoder_output = self.encoder(sample["warped_dst"])
        dst_inter_B_output = self.inter_B(dst_encoder_output)
        dst_inter_AB_output = self.inter_AB(dst_encoder_output)
        dst_inter_B_AB_concat = torch.cat((dst_inter_B_output, dst_inter_AB_output), dim=1)
        dst_inter_AB_concat = torch.cat((dst_inter_AB_output, dst_inter_AB_output), dim=1)

        src_src_pred, src_src_mask_pred = self.decoder(src_inter_AB_concat)
        dst_dst_pred, dst_dst_mask_pred = self.decoder(dst_inter_B_AB_concat)
        src_dst_pred, src_dst_mask_pred = self.decoder(dst_inter_AB_concat)

        # unpack masks from one combined mask
        target_src_mask = torch.clip(sample["target_src_mask"], 0, 1)
        target_dst_mask = torch.clip(sample["target_dst_mask"], 0, 1)
        target_src_mask_eyes = torch.clip(sample["target_src_mask"] - 1, 0, 1)
        target_dst_mask_eyes = torch.clip(sample["target_dst_mask"] - 1, 0, 1)

        target_src_mask_blur = gaussian_blur(target_src_mask, max(1, self.resolution // 32))
        target_src_mask_blur = torch.clip(target_src_mask_blur, 0, 0.5) * 2

        target_dst_mask_blur = gaussian_blur(target_dst_mask, max(1, self.resolution // 32))
        target_dst_mask_blur = torch.clip(target_dst_mask_blur, 0, 0.5) * 2

        target_dst_masked = sample["target_dst"] * target_dst_mask_blur
        target_src_masked = sample["target_src"] * target_src_mask_blur

        target_src_masked_opt = target_src_masked if self.masked_training else sample["target_src"]
        target_dst_masked_opt = target_dst_masked if self.masked_training else sample["target_dst"]

        src_src_pred_masked_opt = src_src_pred * target_src_mask_blur if self.masked_training else src_src_pred
        dst_dst_pred_masked_opt = dst_dst_pred * target_dst_mask_blur if self.masked_training else dst_dst_pred

        src_loss = self.compute_loss(
            target=target_src_masked_opt,
            prediction=src_src_pred_masked_opt,
            target_mask=target_src_mask,
            predicted_mask=src_src_mask_pred,
            target_eyes_mask=target_src_mask_eyes
        )
        dst_loss = self.compute_loss(
            target=target_dst_masked_opt,
            prediction=dst_dst_pred_masked_opt,
            target_mask=target_dst_mask,
            predicted_mask=dst_dst_mask_pred,
            target_eyes_mask=target_dst_mask_eyes
        )

        G_loss = src_loss + dst_loss
        G_loss.mean().backward()

        self.encoder_optimiser.step()
        self.inter_B_optimiser.step()
        self.inter_AB_optimiser.step()
        self.decoder_optimiser.step()

        return src_loss, dst_loss, G_loss
