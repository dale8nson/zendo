import torch
from safetensors.torch import load_file, save_file
import argparse
import os
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

tok = CLIPTokenizer.from_pretrained('../models/sdxl-base-1.0', subfolder='tokenizer')

tok2 = CLIPTokenizer.from_pretrained('../models/sdxl-base-1.0', subfolder='tokenizer_2')

enc = CLIPTextModel.from_pretrained('../models/sdxl-base-1.0', subfolder='text_encoder')

enc2 = CLIPTextModelWithProjection.from_pretrained('../models/sdxl-base-1.0', subfolder='text_encoder_2')

def main():
    global tok, tok2, enc, enc2
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str)
    parser.add_argument('-c', type=str, default='default')
    parser.add_argument('-t', type=str)
    parser.add_argument('-s', type=float, default=1)

    args = parser.parse_args()
    
    path = os.path.join(os.getcwd(), f'../models/user/{args.c}/{args.c}.safetensors')
    path2 = os.path.join(os.getcwd(), f'../models/user/{args.c}/{args.c}_2.safetensors')
    
    tensors_1 = {}
    tensors_2 = {}
    
    if os.path.exists(path):

        tensors_1 = load_file(os.path.join(os.getcwd(), f'../models/user/{args.c}/{args.c}.safetensors'))
        tensors_2 = load_file(os.path.join(os.getcwd(), f'../models/user/{args.c}/{args.c}_2.safetensors'))

    try:
        f = load_file(args.f)

    except:
        print('Unable to load safetensors. Attempting recovery...')
        import recover
        try:
            recover.main(args.f)
        except Exception as e:
            print(e)
            return

        print('safetensors recovered')
        f = load_file('./recovered.safetensors')


    t = args.t.strip()
    
    with torch.no_grad():
        embedding_1 = f['embedding_1']
        embedding_2 = f['embedding_2']
        # enc1
        w1 = enc.get_input_embeddings().weight
        mu1 = w1.norm(dim=1).mean()          # avg norm (exclude the last row if your token is last; adjust if not)
        tid = tok.convert_tokens_to_ids(args.t)
        embedding_1.mul_( (mu1 / w1[tid].norm().clamp_min(1e-6)) * args.s )
        # embedding_1.mul_(args.s )


        # enc2
        w2 = enc2.get_input_embeddings().weight
        mu2 = w2.norm(dim=1).mean()
        tid2 = tok2.convert_tokens_to_ids(args.t)
        embedding_2.mul_( (mu2 / w2[tid2].norm().clamp_min(1e-6)) * args.s )
        # embedding_2.mul_(args.s )


    tensors_1[t] = embedding_1
    tensors_2[t] = embedding_2

    c = args.c.strip()

    save_file(tensors_1, os.path.join(os.getcwd(), f'../models/user/{c}/{c}.safetensors'))
    save_file(tensors_2, os.path.join(os.getcwd(), f'../models/user/{c}/{c}_2.safetensors'))

if __name__ == '__main__':
    main()
