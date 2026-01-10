import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

sch = DDPMScheduler.from_pretrained('../models/sdxl-base-1.0', subfolder='scheduler')

vae = AutoencoderKL.from_pretrained('../models/sdxl-base-1.0', subfolder='vae')

unet = UNet2DConditionModel.from_pretrained('../models/sdxl-base-1.0', subfolder='unet')

tok = CLIPTokenizer.from_pretrained('../models/sdxl-base-1.0', subfolder='tokenizer')

tok2 = CLIPTokenizer.from_pretrained('../models/sdxl-base-1.0', subfolder='tokenizer_2')

enc = CLIPTextModel.from_pretrained('../models/sdxl-base-1.0', subfolder='text_encoder')

enc2 = CLIPTextModelWithProjection.from_pretrained('../models/sdxl-base-1.0', subfolder='text_encoder_2')
