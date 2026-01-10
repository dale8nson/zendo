import torch
from torch import Tensor
import torchvision.transforms as T
from safetensors.torch import save_file, load_file
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from typing import Optional
import os
import math
import json
from pickle import dumps
import argparse
import asyncio
from tqdm.auto import tqdm

class CUDAData:
    def __init__(self, model_path:str, input_dir:str, output_dir:str, token: str, initializer_token: str, output_filename:str, tokenizer: Optional[CLIPTokenizer] = None, tokenizer_2: Optional[CLIPTokenizer] = None, text_encoder: Optional[CLIPTextModel] = None, text_encoder_2: Optional[CLIPTextModelWithProjection] = None, vae: Optional[AutoencoderKL] = None, scheduler: Optional[DDPMScheduler] = None, target_size:int = 1024, filename: Optional[str] = None, seed: Optional[int] = None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.output_filename = output_filename

        if tokenizer is None:
            self.tok = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer')
        else: self.tok = tokenizer

        if tokenizer_2 is None:
            self.tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2')
        else: self.tok2 = tokenizer_2

        if text_encoder is None:
            self.enc = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder', torch_dtype=self.dtype, use_safetensors=True)
        else: self.enc = text_encoder

        self.enc.to(self.device, dtype=self.dtype)

        if text_encoder_2 is None:
            self.enc2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder='text_encoder_2', torch_dtype=self.dtype, use_safetensors=True)
        else: self.enc2 = text_encoder_2

        self.enc2.to(self.device, dtype=self.dtype)

        self.enc.text_model.encoder.requires_grad_(False)
        self.enc.text_model.final_layer_norm.requires_grad_(False)
        self.enc.text_model.embeddings.position_embedding.requires_grad_(False)

        self.enc2.text_model.encoder.requires_grad_(False)
        self.enc2.text_model.final_layer_norm.requires_grad_(False)
        self.enc2.text_model.embeddings.position_embedding.requires_grad_(False)

        if vae is None:
            self.vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
        else: self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=self.dtype)

        if scheduler is None:
            self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
        else: self.scheduler = scheduler

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.token = token
        self.init_token = initializer_token
        self.size = target_size
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.filename = filename
        self.seed = seed

    @torch.enable_grad()
    def process(self):
        tok, tok2, enc, enc2, token, output_dir, size, device, dtype = self.tok, self.tok2, self.enc, self.enc2, self.token, self.input_dir, self.size, self.device, self.dtype

        # process initializer tokens
        
        token_embedding = None
        token_embedding_2 = None

        f1 = self.input_dir.split('/')[-1] + '.safetensors'
        f2 = self.input_dir.split('/')[-1] + '_2.safetensors'
        if os.path.exists(os.path.join(self.input_dir, f1)) and os.path.exists(os.path.join(self.input_dir, f2)):

            tensors_1 = load_file(os.path.join(self.input_dir, f1))
            tensors_2 = load_file(os.path.join(self.input_dir, f2))

            token_embedding = tensors_1.get(self.init_token)
            token_embedding_2 = tensors_2.get(self.init_token)

        if token_embedding is None:
            concept_tokens = tok.tokenize(self.init_token)
            concept_ids= tok.encode(concept_tokens, add_special_tokens=False)

            concept_tokens_2 = tok2.tokenize(self.init_token)
            concept_ids_2 = tok2.encode(concept_tokens_2, add_special_tokens=False)

            token_embedding = torch.zeros(size=(1, enc.config.hidden_size), dtype=self.dtype, device=self.device).squeeze(0)

            token_embedding_2 = torch.zeros(size=(1, enc2.config.hidden_size), dtype=torch.float32, device=self.device).squeeze(0)

            input_embeddings = enc.get_input_embeddings().weight.data
            input_embeddings.requires_grad_(True)
            input_embeddings_2 = enc2.get_input_embeddings().weight.data
            input_embeddings_2.requires_grad_(True)

            for tk, tk2 in zip(concept_ids, concept_ids_2):
                token_embedding = token_embedding + 1 / len(concept_ids) * input_embeddings[tk]
                token_embedding_2 = token_embedding_2 + 1 / len(concept_ids_2) * input_embeddings_2[tk2]

        # process token
        tok.add_tokens(token)
        tok2.add_tokens(token)
        enc.resize_token_embeddings(len(tok))
        enc2.resize_token_embeddings(len(tok2))

        token_id = tok.convert_tokens_to_ids(token)
        token_id_2 = tok2.convert_tokens_to_ids(token)
        self.token_embedding = token_embedding.clone()
        self.token_embedding_2 = token_embedding_2.clone()

        input_embeddings = enc.get_input_embeddings().weight.data
        input_embeddings_2 = enc2.get_input_embeddings().weight.data

        input_embeddings[token_id] = token_embedding.clone().to(device)
        input_embeddings_2[token_id_2] = token_embedding_2.clone().to(device)

        # process prompts
        captions_path = os.path.join(output_dir, 'captions.json')
        with open(captions_path) as f:
            captions = json.load(f)

        data_dir = os.path.join(output_dir, f'datasets/{token}')

        filenames = [self.filename] if self.filename is not None else [f for f in filter(lambda n: n in captions, os.listdir(data_dir))]

        prompts = [captions[f]['caption'] for f in filenames]
        
        steps = len(prompts) + len(filenames)

        progress_bar = tqdm(
            range(0, steps),
            initial=0,
            desc="Progress"
        )

        text_embeds = torch.empty(size=(0, 1280)).to(device)

        prompt_embeddings = torch.empty(size=(0, tok.model_max_length, enc.config.hidden_size + enc2.config.hidden_size), dtype=torch.float32, device=self.device)

        # --- after you've added the placeholder token and resized embeddings ---
        token_id  = tok.convert_tokens_to_ids(token)
        token_id2 = tok2.convert_tokens_to_ids(token)
        
        pids1 = []
        pids2 = []
        am1 = []
        am2 = []

        for prompt in prompts:
            ids1 = tok(prompt, padding='max_length', truncation=True,
                       max_length=tok.model_max_length, return_tensors='pt').input_ids.to(self.device)  # [1,L1]
            pids1.append(ids1)
            ids2 = tok2(prompt, padding='max_length', truncation=True,
                        max_length=tok2.model_max_length, return_tensors='pt').input_ids.to(self.device)  # [1,L2]
            pids2.append(ids2)
            am1.append((ids1 != tok.pad_token_id).long())
            am2.append((ids2 != tok2.pad_token_id).long())
            
            progress_bar.update(1)
            
        self.pids1 = torch.cat(pids1, dim=0).to("cpu", torch.int64)
        self.pids2 = torch.cat(pids2, dim=0).to("cpu", torch.int64)
        self.am1   = torch.cat(am1,   dim=0).to("cpu", torch.int64)
        self.am2   = torch.cat(am2,   dim=0).to("cpu", torch.int64)
        
        print(f'pids1: {self.pids1.shape} pids2: {self.pids2.shape}')
        
        # process images

        timestep_list = []
        targets = []
        image_embeds = []

        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            image = Image.open(file_path).convert('RGB')
            scale_x = size / image.width
            scale_y = size / image.height
            scale = max(scale_x, scale_y)
            image = image.resize((int(math.ceil(image.width * scale)), int(math.ceil(image.height * scale))))
            x1, y1, x2, y2 = [int(math.ceil(n * scale)) for n in captions[filename]['bbox']]
            w, h = x2 - x1, y2 - y1
            x1 = min(x1 + w // 2 - size // 2, int(math.ceil(image.width - size)))
            y1 = min(y1 + h // 2 - size // 2, int(math.ceil(image.height - size)))
            x2 = x1 + size
            y2 = y1 + size
            image = image.crop((x1, y1, x2, y2))
            image = self.flip(image)

            # generate latent

            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            pixel_values = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

            if self.seed is not None:
                torch.manual_seed(self.seed)
                
            latent = self.vae.encode(pixel_values).latent_dist.sample().to(self.device, dtype=self.dtype)
            latent = latent * self.vae.config.scaling_factor
            eps    = torch.randn_like(latent)
            t      = torch.randint(self.scheduler.config.num_train_timesteps, (1,), device=latent.device, dtype=torch.int64)
            print(f'timesteps: {t.shape}')
            noisy  = self.scheduler.add_noise(latent, eps, t)

            image_embeds.append(noisy)
            targets.append(eps)
            timestep_list.append(t)
            
            progress_bar.update(1)

        self.image_embeds = torch.cat(image_embeds, dim=0).to(torch.float16)     # [N,4,128,128]
        self.targets= torch.cat(targets, dim=0).to(torch.float16)          # [N,4,128,128]
        self.timesteps = torch.cat(timestep_list, dim=0).cpu()

        original_size = (size, size)
        target_size = (size, size)
        crop_top_left = (0, 0)

        self.add_time_ids = torch.cat([torch.tensor([original_size + crop_top_left + target_size]) for _ in range(len(filenames))])


    def save(self):

        print(f'pids1.shape: {self.pids1.shape} pids2.shape: {self.pids2.shape}')

        tensors = {
            "0": self.token_embedding.to("cpu", torch.float16),      
            "1": self.token_embedding_2.to("cpu", torch.float16),
            "2": self.pids1,                
            "3": self.pids2, 
            "4": self.am1,
            "5": self.am2,               
            "8": self.image_embeds.to("cpu", torch.float16),
            "9": self.timesteps.to("cpu", torch.int64),       
            "10": self.targets.to("cpu", torch.float16),
            "11": self.add_time_ids.to("cpu", torch.int64),
        }

        save_file(tensors, os.path.join(self.output_dir, self.output_filename))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', type=str, default='default')
    parser.add_argument('-t', type=str)
    parser.add_argument('-i', type=str, default='photo')
    parser.add_argument('-v', type=str)
    parser.add_argument('-o', type=str, default='input.safetensors')
    parser.add_argument('-f', type=str)
    parser.add_argument('-s', type=int)

    args = parser.parse_args()

    c = args.c.strip()

    cwd = os.getcwd()
    model_path = os.path.join(cwd, '../../../models/sdxl-base-1.0')
    input_dir = os.path.join(cwd, f'../../../models/user/{c}')
    output_dir = os.path.join(cwd, '../../../cuda')

    data = CUDAData(model_path=model_path, input_dir=input_dir, output_dir=output_dir, output_filename=args.o, token=args.t, initializer_token=args.i, filename=args.f, seed=args.s)

    data.process()
    data.save()


if __name__ == '__main__':
    main()
