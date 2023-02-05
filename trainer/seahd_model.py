import torch
import torch.optim as optim
import torchshow as ts
from torchmetrics import StructuralSimilarityIndexMeasure
from trainer.deepfake_architecture import Encoder, Decoder, Inter
from trainer.utils import gaussian_blur
from trainer.base_model import BaseModelABC
from params import Params


class SAEHDModel(BaseModelABC):
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

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.encoder = Encoder(in_ch=self.input_ch, e_ch=self.e_dims)
        encoder_out_ch = self.encoder.get_output_length(
            input_resolution=self.resolution
        )

        self.inter_AB = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            lowest_dense_res=self.resolution // 16,
        )
        self.inter_B = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            lowest_dense_res=self.resolution // 16,
        )

        inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        inters_out_ch = inter_AB_out_ch + inter_B_out_ch
        self.decoder = Decoder(
            in_ch=inters_out_ch, d_ch=self.d_dims, d_mask_ch=self.d_mask_dims
        )

        self.encoder_optimiser = optim.Adam(self.encoder.parameters())
        self.inter_B_optimiser = optim.Adam(self.inter_B.parameters())
        self.inter_AB_optimiser = optim.Adam(self.inter_AB.parameters())
        self.decoder_optimiser = optim.Adam(self.decoder.parameters())

    def compute_ssim_loss(self, target, prediction):
        return 1 - self.ssim(prediction, target)

    def compute_loss(
        self, target, prediction, target_mask, predicted_mask, target_eyes_mask
    ):
        if self.resolution < 256:
            loss = 10 * self.compute_ssim_loss(target, prediction)
        else:
            # TODO maybe change function to include kernel size here. Second loss function use smaller kernel size
            loss = 5 * self.compute_ssim_loss(target, prediction)
            loss += 5 * self.compute_ssim_loss(target, prediction)

        loss += torch.mean(10 * torch.square(target - prediction), dim=(1, 2, 3))

        if self.eyes_priority:
            loss += torch.mean(
                input=300
                * torch.abs(target * target_eyes_mask - prediction * target_eyes_mask),
                dim=(1, 2, 3),
            )

        loss += torch.mean(
            10 * torch.square(target_mask - predicted_mask), dim=(1, 2, 3)
        )
        return loss

    def train(self, sample):
        self.encoder.zero_grad()
        self.inter_B.zero_grad()
        self.inter_AB.zero_grad()
        self.decoder.zero_grad()

        src_encoder_output = self.encoder(sample[Params.warped_src])
        src_inter_AB_output = self.inter_AB(src_encoder_output)
        src_inter_AB_concat = torch.cat(
            (src_inter_AB_output, src_inter_AB_output), dim=1
        )

        dst_encoder_output = self.encoder(sample[Params.warped_dst])
        dst_inter_B_output = self.inter_B(dst_encoder_output)
        dst_inter_AB_output = self.inter_AB(dst_encoder_output)
        dst_inter_B_AB_concat = torch.cat(
            (dst_inter_B_output, dst_inter_AB_output), dim=1
        )
        dst_inter_AB_concat = torch.cat(
            (dst_inter_AB_output, dst_inter_AB_output), dim=1
        )

        src_src_pred, src_src_mask_pred = self.decoder(src_inter_AB_concat)
        dst_dst_pred, dst_dst_mask_pred = self.decoder(dst_inter_B_AB_concat)
        src_dst_pred, src_dst_mask_pred = self.decoder(dst_inter_AB_concat)

        # unpack masks from one combined mask
        target_src_mask = torch.clip(sample[Params.target_src_mask], 0, 1)
        target_dst_mask = torch.clip(sample[Params.target_dst_mask], 0, 1)
        target_src_mask_eyes = torch.clip(sample[Params.target_src_mask] - 1, 0, 1)
        target_dst_mask_eyes = torch.clip(sample[Params.target_dst_mask] - 1, 0, 1)

        target_src_mask_blur = gaussian_blur(
            target_src_mask, max(1, self.resolution // 32)
        )
        target_src_mask_blur = torch.clip(target_src_mask_blur, 0, 0.5) * 2

        target_dst_mask_blur = gaussian_blur(
            target_dst_mask, max(1, self.resolution // 32)
        )
        target_dst_mask_blur = torch.clip(target_dst_mask_blur, 0, 0.5) * 2

        target_dst_masked = sample[Params.target_dst] * target_dst_mask_blur
        target_src_masked = sample[Params.target_src] * target_src_mask_blur

        target_src_masked_opt = (
            target_src_masked if self.masked_training else sample[Params.target_src]
        )
        target_dst_masked_opt = (
            target_dst_masked if self.masked_training else sample[Params.target_dst]
        )

        src_src_pred_masked_opt = (
            src_src_pred * target_src_mask_blur
            if self.masked_training
            else src_src_pred
        )
        dst_dst_pred_masked_opt = (
            dst_dst_pred * target_dst_mask_blur
            if self.masked_training
            else dst_dst_pred
        )

        src_loss = self.compute_loss(
            target=target_src_masked_opt,
            prediction=src_src_pred_masked_opt,
            target_mask=target_src_mask,
            predicted_mask=src_src_mask_pred,
            target_eyes_mask=target_src_mask_eyes,
        )
        dst_loss = self.compute_loss(
            target=target_dst_masked_opt,
            prediction=dst_dst_pred_masked_opt,
            target_mask=target_dst_mask,
            predicted_mask=dst_dst_mask_pred,
            target_eyes_mask=target_dst_mask_eyes,
        )

        G_loss = src_loss + dst_loss
        G_loss.mean().backward()

        self.encoder_optimiser.step()
        self.inter_B_optimiser.step()
        self.inter_AB_optimiser.step()
        self.decoder_optimiser.step()

        return src_loss, dst_loss, G_loss

    def save(self, path):
        # use of MLFlow or similar for model tracking and saving
        pass

    def load(self, path):
        pass
