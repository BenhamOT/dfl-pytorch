import multiprocessing
import operator
import torch
import torch.optim as optim
from trainer.deepfake_architecture import Encoder, Decoder, Inter
from trainer.data_loader import C2DataLoader
from PIL import Image

import numpy as np


class SAEHDModel:

    def __init__(self, settings=None):
        self.settings = settings
        self.resolution = 128  # settings["resolution"]
        self.e_dims = 80  # settings["e_dims"]
        self.ae_dims = 128  # settings["ae_dims"]
        self.d_dims = 48  # settings["d_dims"]
        self.d_mask_dims = 16  # settings["d_mask_dims"]
        # self.masked_training = settings["masked_training"]
        # self.learn_mask = settings["learn_mask"]
        # self.eyes_priority = settings["eyes_priority"]
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

    def on_initialize(self):

        # self.model_data_format = "NCHW"
        #
        # if self.pretrain:
        #     self.random_warp = False
        #     self.uniform_yaw = True
        #     self.eyes_prio = False
        #     self.uniform_yaw = True
        #
        # if self.ct_mode == 'none':
        #     self.ct_mode = None

        input_ch=3
        self.model_filename_list = []

        self.encoder = Encoder(in_ch=input_ch, e_ch=self.e_dims)
        encoder_out_ch = 40960  # this is wrong need to change this

        self.inter_AB = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims*2,
            resolution=self.resolution
        )
        self.inter_B = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims*2,
            resolution=self.resolution
        )

        # TODO need to make sure 3 is ok for the inter_AB_(B)_out_ch
        inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        inters_out_ch = inter_AB_out_ch + inter_B_out_ch
        print("Inter combined output channels is {}".format(inters_out_ch))
        self.decoder = Decoder(in_ch=inters_out_ch, d_ch=self.d_dims, d_mask_ch=self.d_mask_dims)

        if self.is_training:

            # Initialize optimizers
            lr=5e-5
            # lr_dropout = 0.3 if self.lr_dropout and not self.pretrain else 1.0
            # clipnorm = 1.0 if self.clip_gradients else 0.0
            # self.src_dst_opt = optim.RMSprop(lr=lr)

            for sample in C2DataLoader().run():
                print("input shape is {}".format(sample["src"].shape))
                test_sample = np.moveaxis(np.array(sample["src"])[0], 0, -1)
                img = Image.fromarray(test_sample, mode="RGB")
                img.save("test_sample.jpg")
                break
                # gpu_src_code = self.encoder(sample["src"])
                # print(gpu_src_code.shape)
                # gpu_src_inter_AB_code = self.inter_AB(gpu_src_code)
                # print("inter_AB output shape is {}".format(gpu_src_inter_AB_code.shape))
                # gpu_src_code = torch.cat((gpu_src_inter_AB_code, gpu_src_inter_AB_code), dim=1)
                # print("The shape of the gpu_src_code after is concatenated is {}".format(gpu_src_code.shape))
                # gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                # print("The gpu_pred_src_src output shape is {}".format(gpu_pred_src_src.shape))
                # print("The gpu_pred_src_srcm output shape is {}".format(gpu_pred_src_srcm.shape))
                # break

        # if self.is_training:
        #
        #     # Compute losses per GPU
        #     gpu_pred_src_src_list = []
        #     gpu_pred_dst_dst_list = []
        #     gpu_pred_src_dst_list = []
        #     gpu_pred_src_srcm_list = []
        #     gpu_pred_dst_dstm_list = []
        #     gpu_pred_src_dstm_list = []
        #
        #     gpu_src_losses = []
        #     gpu_dst_losses = []
        #     gpu_G_loss_gvs = []
        #     gpu_D_code_loss_gvs = []
        #     gpu_D_src_dst_loss_gvs = []
        #     for gpu_id in range(gpu_count):
        #         with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
        #
        #             with tf.device(f'/CPU:0'):
        #                 # slice on CPU, otherwise all batch data will be transfered to GPU first
        #                 batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
        #                 gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
        #                 gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
        #                 gpu_target_src      = self.target_src [batch_slice,:,:,:]
        #                 gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
        #                 gpu_target_srcm_all = self.target_srcm_all[batch_slice,:,:,:]
        #                 gpu_target_dstm_all = self.target_dstm_all[batch_slice,:,:,:]
        #
        #             # process model tensors
        #             gpu_src_code = self.encoder (gpu_warped_src)
        #             gpu_src_inter_AB_code = self.inter_AB (gpu_src_code)
        #             gpu_src_code = tf.concat([gpu_src_inter_AB_code,gpu_src_inter_AB_code], nn.conv2d_ch_axis  )
        #             gpu_dst_code = self.encoder (gpu_warped_dst)
        #             gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
        #             gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
        #             gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )
        #             gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )
        #
        #             gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
        #             gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
        #             gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
        #
        #             gpu_pred_src_src_list.append(gpu_pred_src_src)
        #             gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
        #             gpu_pred_src_dst_list.append(gpu_pred_src_dst)
        #
        #             gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
        #             gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
        #             gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)
        #
        #             # unpack masks from one combined mask
        #             gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)
        #             gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)
        #             gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_all-1, 0, 1)
        #             gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_all-1, 0, 1)
        #
        #             gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
        #             gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2
        #
        #             gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )
        #             gpu_target_dstm_style_blur = gpu_target_dstm_blur #default style mask is 0.5 on boundary
        #             gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2
        #
        #             gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
        #             gpu_target_dst_style_masked      = gpu_target_dst*gpu_target_dstm_style_blur
        #             gpu_target_dst_style_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_style_blur)
        #
        #             gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
        #             gpu_target_dst_masked_opt  = gpu_target_dst_masked if masked_training else gpu_target_dst
        #
        #             gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
        #             gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst
        #
        #             gpu_psd_target_dst_style_masked = gpu_pred_src_dst*gpu_target_dstm_style_blur
        #             gpu_psd_target_dst_style_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_style_blur)
        #
        #             if resolution < 256:
        #                 gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
        #             else:
        #                 gpu_src_loss =  tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
        #                 gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
        #             gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
        #
        #             if eyes_prio:
        #                 gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_srcm_eyes - gpu_pred_src_src*gpu_target_srcm_eyes ), axis=[1,2,3])
        #
        #             gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )
        #
        #             if resolution < 256:
        #                 gpu_dst_loss = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
        #             else:
        #                 gpu_dst_loss = tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
        #                 gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])
        #             gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
        #
        #             if eyes_prio:
        #                 gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_dstm_eyes - gpu_pred_dst_dst*gpu_target_dstm_eyes ), axis=[1,2,3])
        #
        #             gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )
        #
        #             gpu_src_losses += [gpu_src_loss]
        #             gpu_dst_losses += [gpu_dst_loss]
        #
        #             gpu_G_loss = gpu_src_loss + gpu_dst_loss
        #
        #             def DLoss(labels,logits):
        #                 return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])
        #
        #             gpu_G_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]
        #
        #
        #     # Average losses and gradients, and create optimizer update ops
        #     with tf.device (models_opt_device):
        #         pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
        #         pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
        #         pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
        #         pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
        #         pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
        #         pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)
        #
        #         src_loss = tf.concat(gpu_src_losses, 0)
        #         dst_loss = tf.concat(gpu_dst_losses, 0)
        #         src_dst_loss_gv_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))
        #
        #     # Initializing training and view functions
        #     def src_dst_train(warped_src, target_src, target_srcm_all, \
        #                       warped_dst, target_dst, target_dstm_all):
        #         s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
        #                                     feed_dict={self.warped_src :warped_src,
        #                                                self.target_src :target_src,
        #                                                self.target_srcm_all:target_srcm_all,
        #                                                self.warped_dst :warped_dst,
        #                                                self.target_dst :target_dst,
        #                                                self.target_dstm_all:target_dstm_all,
        #                                                })
        #         return s, d
        #     self.src_dst_train = src_dst_train
        #
        #
        #     def AE_view(warped_src, warped_dst):
        #         return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
        #                                     feed_dict={self.warped_src:warped_src,
        #                                             self.warped_dst:warped_dst})
        #     self.AE_view = AE_view


SAEHDModel().on_initialize()
