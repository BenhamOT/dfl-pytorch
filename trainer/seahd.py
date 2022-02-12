import multiprocessing
import operator
from trainer.deepfake_architecture import Encoder, Decoder, Inter

import numpy as np


class SAEHDModel:

    def __init__(self, settings):

        self.resolution = settings["resolution"]
        self.ae_dims = settings["ae_dims"]
        self.e_dims = settings["e_dims"]
        self.d_dims = settings["d_dims"]
        self.d_mask_dims = settings["d_mask_dims"]
        self.masked_training = settings["masked_training"]
        self.learn_mask = settings["learn_mask"]
        self.eyes_priority = settings["eyes_priority"]
        self.lr_dropout = settings["lr_dropout"]
        self.random_warp = settings["random_warp"]  # want to be false for pretraining
        self.target_iterations = settings["number_of_iterations"]
        self.random_flip = settings["random_flip"]
        self.batch_size = settings["batch_size"]
        self.pretrain = settings["pretrain"]
        self.uniform_yaw = settings["uniform_yaw"]
        self.ct_mode = settings["ct_mode"]
        self.clip_gradients = settings["clip_gradients"]

    def on_initialize(self):

        self.model_data_format = "NCHW"

        if self.pretrain:
            self.random_warp = False
            self.uniform_yaw = True
            self.eyes_prio = False
            self.uniform_yaw = True

        if self.ct_mode == 'none':
            self.ct_mode = None

        input_ch=3
        bgr_shape = nn.get4Dshape(self.resolution,self.resolution,input_ch)
        self.model_filename_list = []

        self.encoder = Encoder(in_ch=input_ch, e_ch=self.e_dims)
        encoder_out_ch = self.encoder.compute_output_channels ( (nn.floatx, bgr_shape))

        self.inter_AB = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims*2,
            resolution=self.resolution
        )  #  name='inter_AB')

        self.inter_B = Inter(
            in_ch=encoder_out_ch,
            ae_ch=self.ae_dims,
            ae_out_ch=self.ae_dims*2,
            resolution=self.resolution
        )  # name='inter_B')

        inter_AB_out_ch = self.inter_AB.compute_output_channels((nn.floatx, (None,encoder_out_ch)))
        inter_B_out_ch = self.inter_B.compute_output_channels ( (nn.floatx, (None,encoder_out_ch)))
        inters_out_ch = inter_AB_out_ch+inter_B_out_ch
        self.decoder = Decoder(in_ch=inters_out_ch, d_ch=self.d_dims, d_mask_ch=self.d_mask_dims)

        if self.is_training:

            # Initialize optimizers
            lr=5e-5
            lr_dropout = 0.3 if self.lr_dropout in ['y','cpu'] and not self.pretrain else 1.0
            clipnorm = 1.0 if self.clip_gradients else 0.0


            self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter_AB.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()
            self.src_dst_opt = nn.RMSprop(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables (self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]


        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)


            # Compute losses per GPU
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_G_loss_gvs = []
            gpu_D_code_loss_gvs = []
            gpu_D_src_dst_loss_gvs = []
            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):

                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm_all = self.target_srcm_all[batch_slice,:,:,:]
                        gpu_target_dstm_all = self.target_dstm_all[batch_slice,:,:,:]

                    # process model tensors
                    gpu_src_code = self.encoder (gpu_warped_src)
                    gpu_src_inter_AB_code = self.inter_AB (gpu_src_code)
                    gpu_src_code = tf.concat([gpu_src_inter_AB_code,gpu_src_inter_AB_code], nn.conv2d_ch_axis  )

                    gpu_dst_code = self.encoder (gpu_warped_dst)
                    gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                    gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                    gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )
                    gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )

                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    # unpack masks from one combined mask
                    gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)
                    gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)
                    gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_all-1, 0, 1)
                    gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_all-1, 0, 1)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2

                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt  = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    if resolution < 256:
                        gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    else:
                        gpu_src_loss =  tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])

                    if eyes_prio:
                        gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_srcm_eyes - gpu_pred_src_src*gpu_target_srcm_eyes ), axis=[1,2,3])

                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    if resolution < 256:
                        gpu_dst_loss = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    else:
                        gpu_dst_loss = tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])

                    if eyes_prio:
                        gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_dstm_eyes - gpu_pred_dst_dst*gpu_target_dstm_eyes ), axis=[1,2,3])

                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss

                    def DLoss(labels,logits):
                        return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])

                    gpu_G_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))

            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm_all, \
                              warped_dst, target_dst, target_dstm_all):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm_all:target_srcm_all,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm_all:target_dstm_all,
                                                       })
                return s, d
            self.src_dst_train = src_dst_train


            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src,
                                                    self.warped_dst:warped_dst})
            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):


                gpu_dst_code = self.encoder (self.warped_dst)
                gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)
                gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)


            def AE_merge( warped_dst):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if 'df' in archi_type:
                    if model == self.inter:
                        do_init = True
                elif 'liae' in archi_type:
                    if model == self.inter_AB or model == self.inter_B:
                        do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None and not self.pretrain else None

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

            self.last_src_samples_loss = []
            self.last_dst_samples_loss = []

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
               (io.is_colab() and self.iter % 100 == 0)

    #override
    def onTrainOneIter(self):
        if self.get_iter() == 0 and not self.pretrain and not self.pretrain_just_disabled:
            io.log_info('You are training the model from scratch. It is strongly recommended to use a pretrained model to speed up the training and improve the quality.\n')

        bs = self.get_batch_size()

        ( (warped_src, target_src, target_srcm_all), \
          (warped_dst, target_dst, target_dstm_all) ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm_all, warped_dst, target_dst, target_dstm_all)

        for i in range(bs):
            self.last_src_samples_loss.append (  (target_src[i], target_srcm_all[i], src_loss[i] )  )
            self.last_dst_samples_loss.append (  (target_dst[i], target_dstm_all[i], dst_loss[i] )  )

        if len(self.last_src_samples_loss) >= bs*16:
            src_samples_loss = sorted(self.last_src_samples_loss, key=operator.itemgetter(2), reverse=True)
            dst_samples_loss = sorted(self.last_dst_samples_loss, key=operator.itemgetter(2), reverse=True)

            target_src      = np.stack( [ x[0] for x in src_samples_loss[:bs] ] )
            target_srcm_all = np.stack( [ x[1] for x in src_samples_loss[:bs] ] )

            target_dst      = np.stack( [ x[0] for x in dst_samples_loss[:bs] ] )
            target_dstm_all = np.stack( [ x[1] for x in dst_samples_loss[:bs] ] )

            src_loss, dst_loss = self.src_dst_train (target_src, target_src, target_srcm_all, target_dst, target_dst, target_dstm_all)
            self.last_src_samples_loss = []
            self.last_dst_samples_loss = []

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples):
        ( (warped_src, target_src, target_srcm_all,),
          (warped_dst, target_dst, target_dstm_all,) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm_all, target_dstm_all = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm_all, target_dstm_all] )]

        target_srcm = np.clip(target_srcm_all, 0, 1)
        target_dstm = np.clip(target_dstm_all, 0, 1)

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        if self.resolution <= 256:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]

                ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('SAEHD masked', np.concatenate (st_m, axis=0 )), ]
        else:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD src-src', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD dst-dst', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD pred', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked src-src', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = D[i]*target_dstm[i], DD[i]*DDM[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked dst-dst', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i]*target_dstm[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked pred', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = SAEHDModel
