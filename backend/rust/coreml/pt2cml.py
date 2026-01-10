import torch
import coremltools as ct

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors.torch import load_file
from collections import namedtuple

from typing import NamedTuple, Dict, Any

import argparse

UnetKwargs = namedtuple('UnetKwargs', ['example_kwarg_inputs', 'added_cond_kwargs'])
example_kwarg_inputs = namedtuple('example_kwarg_inputs', ['sample','timestep', 'encoder_hidden_states', 'attention_mask', 'added_cond_kwargs'])
added_cond_kwargs = namedtuple('add_cond_kwargs', ['text_embeds', 'time_ids'])
parser = argparse.ArgumentParser()

# parser.add_argument('-f', type=str)
parser.add_argument('-m', type=str)
parser.add_argument('-p', type=str)
parser.add_argument('-o', type=str)

model_classes = {
  'vae': {'class': AutoencoderKL, 'shape': ((1, 3, 1024, 1024))},
  'unet': {'class': UNet2DConditionModel, 
          #  'example_kwarg_inputs': example_kwarg_inputs(
          #   sample = torch.randn(1, 4, 128, 128), 
          # timestep=torch.randint(1000, (1,), dtype=torch.uint64), encoder_hidden_states=torch.randn(1,77, 2048), attention_mask=torch.randn(1, 77), 
          #   added_cond_kwargs=added_cond_kwargs(text_embeds=torch.randn(1, 1280), 
          #   time_ids=torch.tensor([(1024, 1024) + (0, 0) + (1024, 1024)])))
          'example_kwarg_inputs': {
            'sample': torch.randn(1, 4, 128, 128), 
            'timestep': torch.randint(1000, (1,), dtype=torch.uint64), 'encoder_hidden_states': torch.randn(1, 77, 2048), 'attention_mask': torch.randn(1, 0), 
            'added_cond_kwargs': added_cond_kwargs(text_embeds=torch.randn(1, 1280).to(torch.device('mps')), 
            time_ids=torch.tensor([(1024, 1024) + (0, 0) + (1024, 1024)]).to(torch.device('mps')))}
}, 
  'text_encoder': {'class': CLIPTextModel, 'shape': ((1, 77, 1280)), 'kwargs': {}},
  'text_encoder_2': {'class': CLIPTextModelWithProjection, 'shape': (1, 77, 1280), 'kwargs': {}}
}

params = parser.parse_args()

# files = args.f.split(',')
models = params.m.split(',')
output_files = params.o.split(',')

print(f'models: {len(models)} outputs: {len(output_files)}')

for model_type, output in zip(models, output_files):
  
    # state_dict = load_file(file)
    
    model_class = model_classes[model_type]['class']
    # shape = model_classes[model_type]['shape']
    kwargs = model_classes[model_type]['example_kwarg_inputs']
    
    def to_metal(args):
      for arg in args:
        if isinstance(arg, type(torch.Tensor)):
          arg = arg.to(torch.device('mps'))
        elif isinstance(arg, (type(added_cond_kwargs), dict, list, tuple)):
          to_metal(arg)
        
    to_metal(kwargs) 

    model = model_class.from_pretrained(params.p, subfolder=model_type, use_safetensors=True, torch_dtype=torch.float32)
    
    # model.load_state_dict(state_dict)
    model.eval()
    
    # inp = tuple([torch.randn(s) for s in shape])
  
    # input = torch.randn(*model_classes[model_type]['shape'])
    model = model.to("mps")
    
    traced = torch.jit.trace(model, example_kwarg_inputs=kwargs)
    
    mlmodel = ct.convert(
      traced,
      inputs=[ct.TensorType(shape=s.shape) for s in kwargs],
  )
    
    mlmodel.save(params.o)



