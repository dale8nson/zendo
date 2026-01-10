import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection 
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr

from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from accelerate import Accelerator
from accelerate.utils import set_seed

from safetensors.torch import load_file, save_file
import argparse

import os


class LoRA:
    def __init__(
        self,
        model_path: str,
        data_file:str,
        lr: float=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, gradient_accumulation_steps=1, lr_warmup_steps=20, max_train_steps=2000, batch_size=1, num_batches=1, image_size=1024, checkpoint=0, use_8bit_adam=False, rank=4):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device = device

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.max_train_steps = max_train_steps
        self.size = image_size

        accelerator_project_config = ProjectConfiguration()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="fp16",
            project_config=accelerator_project_config,
        )

        dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            dtype = torch.bfloat16

        self.dtype = dtype
        
        self.num_images = image_embeds.shape[0]

        self.enc = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder')
        self.enc2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder='text_encoder_2')
        
        for p in self.enc.parameters(): p.requires_grad_(False)
        for p in self.enc2.parameters(): p.requires_grad_(False)
        
        if use_8bit_adam:
            
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        
        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        
        self.lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
        
        unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.lr_scheduler
        )

        self.samples = []
        N = image_embeds.shape[0]
        for _ in range(self.num_batches):
            idx = torch.randperm(N).tolist()
            for i in range(0, self.batch_size):
                j = idx[i % N]
                self.samples.append({
                    "latents":    image_embeds[j].unsqueeze(0),
                    "target":     targets[j].unsqueeze(0),               
                    "t":          timesteps[j].unsqueeze(0),             
                    "pids1":         pids1[j].unsqueeze(0),            
                    "pids2":         pids2[j].unsqueeze(0),
                    "am1":           am1[j].unsqueeze(0),
                    "am2":           am2[j].unsqueeze(0),            
                    "time_ids":   time_ids[j].unsqueeze(0),
                })
        
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True)
        self.unet.eval()
        self.unet.enable_gradient_checkpointing()
        self.unet.requires_grad_(False)
        self.unet = self.unet.to(device=self.accelerator.device, dtype=dtype)
        self.unet.requires_grad_(False)

        if is_xformers_available():
            import xformers
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.checkpoint = checkpoint
        
        self.lora_config = LoraConfig(
        r=rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    def train(self):
        
        


def main():
  parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', type=str, default='input.safetensors')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_warmup_steps', type=int, default=20)
    parser.add_argument('--max_train_steps', type=int, default=2000)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)


    args = parser.parse_args()

    trainer = CUDATrainer(
        model_path=args.model_path,
        data_file=args.data_file,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        max_train_steps=args.max_train_steps,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        image_size=args.image_size,
        checkpoint=args.checkpoint,
    )

    trainer.train(global_step=args.start)

    # with open('tensors.safetensors', 'wb') as f:
    #     f.write(learned_embeds)

if __name__ == '__main__':
    main()