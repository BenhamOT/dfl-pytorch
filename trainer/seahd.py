import torch
import torch.optim as optim
import torchshow as ts

from trainer.deepfake_architecture import Encoder, Decoder, Inter
from trainer.data_loader import CustomDataLoader
from trainer.utils import gaussian_blur, dssim


class SAEHDModel:

    def __init__(self, settings=None):
        self.settings = settings
        self.resolution = 128  # settings["resolution"]
        self.e_dims = 80  # settings["e_dims"]
        self.ae_dims = 128  # settings["ae_dims"]
        self.d_dims = 48  # settings["d_dims"]
        self.d_mask_dims = 16  # settings["d_mask_dims"]
        self.masked_training = True  # settings["masked_training"]
        # self.learn_mask = settings["learn_mask"]
        self.eyes_priority = False  # settings["eyes_priority"]
        # self.lr_dropout = settings["lr_dropout"]
        # self.random_warp = settings["random_warp"] # want to be false for pretraining
        # self.target_iterations = settings["number_of_iterations"]
        # self.random_flip = settings["random_flip"]
        # self.batch_size = settings["batch_size"]
        # self.pretrain = settings["pretrain"]
        # self.uniform_yaw = settings["uniform_yaw"]
        # self.ct_mode = settings["ct_mode"]
        # self.clip_gradients = settings["clip_gradients"]
        self.is_training = True
        self.epochs = 1

        input_ch = 3
        self.encoder = Encoder(in_ch=input_ch, e_ch=self.e_dims)
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

        self.encoder_optimiser = optim.RMSprop(self.encoder.parameters())
        self.inter_B_optimiser = optim.RMSprop(self.inter_B.parameters())
        self.inter_AB_optimiser = optim.RMSprop(self.inter_AB.parameters())
        self.decoder_optimiser = optim.RMSprop(self.decoder.parameters())

    def run(self, src_path: str, dst_path: str):

        for i in range(self.epochs):
            print("epoch {}".format(i))

            src_average_losses = []
            dst_losses = []
            G_loss_gvs = []

            # TODO add in tqdm here when finsihed debugging
            for sample in CustomDataLoader(src_path=src_path, dst_path=dst_path).run():
                # TODO need to compare input here to dfl input

                src_loss, dst_loss, combined_loss = self.train(sample=sample)
                src_average_losses.append(src_loss.mean())

                print("src loss is {}".format(src_loss))
                print("dst loss is {}".format(dst_loss))
                print("combined loss is {}".format(combined_loss))
            # sns.lineplot(x=src_average_losses, y=[i for i in range(len(src_average_losses))])
            # plt.show()

    def train(self, sample):

        self.encoder.zero_grad()
        self.inter_B.zero_grad()
        self.inter_AB.zero_grad()
        self.decoder.zero_grad()

        gpu_src_code = self.encoder(sample["warped_src"])
        gpu_src_inter_AB_code = self.inter_AB(gpu_src_code)
        gpu_src_code = torch.cat((gpu_src_inter_AB_code, gpu_src_inter_AB_code), dim=1)

        gpu_dst_code = self.encoder(sample["warped_dst"])
        gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
        gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
        gpu_dst_code = torch.cat((gpu_dst_inter_B_code, gpu_dst_inter_AB_code), dim=1)
        gpu_src_dst_code = torch.cat((gpu_dst_inter_AB_code, gpu_dst_inter_AB_code), dim=1)

        # TODO find a way to visualise the gpu_pred_src_src
        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
        ts.show(gpu_pred_src_src)
        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)

        # unpack masks from one combined mask
        gpu_target_srcm = torch.clip(sample["target_src_mask"], 0, 1)
        gpu_target_dstm = torch.clip(sample["target_dst_mask"], 0, 1)
        gpu_target_srcm_eyes = torch.clip(sample["target_src_mask"] - 1, 0, 1)
        gpu_target_dstm_eyes = torch.clip(sample["target_dst_mask"] - 1, 0, 1)

        gpu_target_srcm_blur = gaussian_blur(gpu_target_srcm, max(1, self.resolution // 32))
        gpu_target_srcm_blur = torch.clip(gpu_target_srcm_blur, 0, 0.5) * 2

        gpu_target_dstm_blur = gaussian_blur(gpu_target_dstm, max(1, self.resolution // 32))
        gpu_target_dstm_blur = torch.clip(gpu_target_dstm_blur, 0, 0.5) * 2

        # TODO how does this multiplication take place?
        gpu_target_dst_masked = sample["target_dst"] * gpu_target_dstm_blur
        gpu_target_src_masked = sample["target_src"] * gpu_target_srcm_blur

        gpu_target_src_masked_opt = gpu_target_src_masked if self.masked_training else sample["target_src"]
        gpu_target_dst_masked_opt = gpu_target_dst_masked if self.masked_training else sample["target_dst"]

        gpu_pred_src_src_masked_opt = gpu_pred_src_src * gpu_target_srcm_blur if self.masked_training else gpu_pred_src_src
        gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst * gpu_target_dstm_blur if self.masked_training else gpu_pred_dst_dst

        if self.resolution < 256:
            dssim_src_result = 10 * dssim(
                img1=gpu_target_src_masked_opt,
                img2=gpu_pred_src_src_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            gpu_src_loss = torch.mean(dssim_src_result, dim=1)
        else:
            dssim_src_result = 5 * dssim(
                img1=gpu_target_src_masked_opt,
                img2=gpu_pred_src_src_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            gpu_src_loss = torch.mean(dssim_src_result, dim=1)
            dssim_src_result = 5 * dssim(
                img1=gpu_target_src_masked_opt,
                img2=gpu_pred_src_src_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 23.2)
            )
            gpu_src_loss += torch.mean(dssim_src_result, dim=1)

        gpu_src_loss += torch.mean(10 * torch.square(gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt),
                                   dim=(1, 2, 3))

        if self.eyes_priority:
            gpu_src_loss += torch.mean(
                input=300 * torch.abs(
                    sample["target_src"] * gpu_target_srcm_eyes - gpu_pred_src_src * gpu_target_srcm_eyes),
                dim=(1, 2, 3)
            )

        gpu_src_loss += torch.mean(10 * torch.square(gpu_target_srcm - gpu_pred_src_srcm), dim=(1, 2, 3))

        if self.resolution < 256:
            dssim_dst_result = 10 * dssim(
                img1=gpu_target_dst_masked_opt,
                img2=gpu_pred_dst_dst_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            gpu_dst_loss = torch.mean(dssim_dst_result, dim=1)
        else:
            dssim_dst_result = 5 * dssim(
                img1=gpu_target_dst_masked_opt,
                img2=gpu_pred_dst_dst_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 11.6)
            )
            gpu_dst_loss = torch.mean(dssim_dst_result, dim=1)
            dssim_dst_result = 5 * dssim(
                img1=gpu_target_dst_masked_opt,
                img2=gpu_pred_dst_dst_masked_opt,
                max_val=1.0,
                filter_size=int(self.resolution / 23.2)
            )
            gpu_dst_loss += torch.mean(dssim_dst_result, dim=1)

        gpu_dst_loss += torch.mean(10 * torch.square(gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt),
                                   dim=(1, 2, 3))

        if self.eyes_priority:
            gpu_dst_loss += torch.mean(
                input=300 * torch.abs(
                    sample["target_dst"] * gpu_target_dstm_eyes - gpu_pred_dst_dst * gpu_target_dstm_eyes),
                dim=(1, 2, 3)
            )

        gpu_dst_loss += torch.mean(10 * torch.square(gpu_target_dstm - gpu_pred_dst_dstm), dim=(1, 2, 3))

        gpu_G_loss = gpu_src_loss + gpu_dst_loss
        gpu_G_loss.mean().backward()

        self.encoder_optimiser.step()
        self.inter_B_optimiser.step()
        self.inter_AB_optimiser.step()
        self.decoder_optimiser.step()

        return gpu_src_loss, gpu_dst_loss, gpu_G_loss


SAEHDModel().run(src_path="../workspace/data_src/", dst_path="../workspace/data_dst/")

#%%
