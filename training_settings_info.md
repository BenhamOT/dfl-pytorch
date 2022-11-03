ae_dims: Autoencoder dimensions - 
    All face information will packed to AE dims. 
    If amount of AE dims are not enough, then for example closed eyes will not be recognized.
    More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU.
e_dims: Encoder dimensiobs - 
    More dims help to recognize more facial features and achieve sharper result, but require more VRAM.
d_dims: Decoder dimensions - 
    More dims help to recognize more facial features and achieve sharper result, but require more VRAM.
d_mask_dims: Decoder mask dimensions - 
    Typical mask dimensions = decoder dimensions / 3. 
    If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality.
masked_training: Masked training - 
    This option is available only for 'whole_face' or 'head' type. 
    Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.