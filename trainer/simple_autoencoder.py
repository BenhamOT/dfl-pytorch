import torch
import torch.optim as optim
import torchshow as ts
from torchmetrics import StructuralSimilarityIndexMeasure
from trainer.deepfake_architecture import Encoder, Decoder, Inter
from trainer.data_loader import CustomDataLoader
from trainer.utils import gaussian_blur
from params import Params


class Autoencoder:
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
        encoder_out_ch = self.encoder.get_output_length(input_resolution=self.resolution)

        self.inter_AB = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            lowest_dense_res=self.resolution // 16
        )
        self.inter_B = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims * 2,
            lowest_dense_res=self.resolution // 16
        )

        inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        inters_out_ch = inter_AB_out_ch + inter_B_out_ch
        self.decoder = Decoder(in_ch=inters_out_ch, d_ch=self.d_dims, d_mask_ch=self.d_mask_dims)

        self.encoder_optimiser = optim.Adam(self.encoder.parameters())
        self.inter_B_optimiser = optim.Adam(self.inter_B.parameters())
        self.inter_AB_optimiser = optim.Adam(self.inter_AB.parameters())
        self.decoder_optimiser = optim.Adam(self.decoder.parameters())

    def run(self, src_path: str, dst_path: str):
        self.encoder.train()
        self.inter_B.train()
        self.inter_AB.train()
        self.decoder.train()

        for i in range(self.epochs):
            print("epoch {}".format(i))

            src_average_losses = []

            # TODO add in tqdm here when finsihed debugging
            for sample in CustomDataLoader(src_path=src_path, dst_path=dst_path).run():
                loss, pred = self.train(sample=sample)
                src_average_losses.append(loss.mean())
                print("loss is {}".format(loss))

            print(f"The average loss for the epoch is {sum(src_average_losses)/len(src_average_losses)}")
            ts.save(pred, f"predictions/epoch_{i}.png")

    def compute_loss(self, target, prediction):
        return 1 - self.ssim(prediction, target)

    def train(self, sample):

        self.encoder.zero_grad()
        self.inter_B.zero_grad()
        self.inter_AB.zero_grad()
        self.decoder.zero_grad()

        src_encoder_output = self.encoder(sample["target_src"])
        src_inter_AB_output = self.inter_AB(src_encoder_output)
        src_inter_AB_concat = torch.cat((src_inter_AB_output, src_inter_AB_output), dim=1)
        src_src_pred, src_src_mask_pred = self.decoder(src_inter_AB_concat)

        # unpack masks from one combined mask
        target_src_mask = torch.clip(sample["target_src_mask"], 0, 1)
        target_src_mask_blur = gaussian_blur(target_src_mask, max(1, self.resolution // 32))
        target_src_masked = sample["target_src"] * target_src_mask_blur
        target_src_masked_opt = target_src_masked if self.masked_training else sample["target_src"]
        src_src_pred_masked_opt = src_src_pred * target_src_mask_blur if self.masked_training else src_src_pred

        src_loss = self.compute_loss(
            target=target_src_masked_opt,
            prediction=src_src_pred_masked_opt,
        )

        G_loss = src_loss
        G_loss.backward()

        self.encoder_optimiser.step()
        self.inter_B_optimiser.step()
        self.inter_AB_optimiser.step()
        self.decoder_optimiser.step()

        return G_loss, src_src_pred_masked_opt
