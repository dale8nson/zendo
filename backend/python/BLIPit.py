#!/usr/bin/env python3
# BLIPit.py
import os, sys, argparse, glob, random
from typing import List
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import contextlib
import torch.nn.functional as F

# HF Transformers
from transformers import (
    BlipForConditionalGeneration, BlipProcessor,
    CLIPModel, CLIPProcessor, AutoConfig
)

# Optional deps: bitsandbytes (8-bit optimizer), peft (LoRA)
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from peft import LoraConfig, get_peft_model
    
    _HAS_PEFT = True
    
except Exception:
    _HAS_PEFT = False


# ---------------------------
# Utilities
# ---------------------------

def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n-1)] + "…"

def _c(s: str, color: str, enabled: bool) -> str:
    if not enabled: return s
    table = {"g":"\033[32m","r":"\033[31m","y":"\033[33m","b":"\033[34m","dim":"\033[2m","/":"\033[0m"}
    return f"{table.get(color,'')}{s}{table['/']}"

def seed_everything(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def print_trainable_params(model, label="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] trainable params: {trainable:,} / {total:,} "
          f"({100.0*trainable/total:.2f}%)")

# Simple image folder dataset (recurses)
class ImageFolderNoLabels(Dataset):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    def __init__(self, root: str):
        self.files = []
        for ext in self.exts:
            self.files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No images found under: {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return img


def normalize_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)


def get_token_logp_sum(gen_scores, sequences) -> torch.Tensor:
    """
    gen_scores: list[T] of logits [B, vocab] for each generated step
    sequences : [B, T+1], includes BOS at position 0 (HF convention)
    returns   : [B] sum of log-probs of sampled tokens
    """
    seq = sequences[:, 1:]  # drop BOS
    logps = []
    for t, scores_t in enumerate(gen_scores):
        # gather log-prob of the actually chosen token at step t
        logp_t = F.log_softmax(scores_t, dim=-1).gather(1, seq[:, t:t+1])
        logps.append(logp_t)
    return torch.cat(logps, dim=1).sum(dim=1)  # [B]

def has_repeat_bigram(text: str) -> float:
    toks = text.lower().split()
    seen = set()
    for i in range(len(toks)-1):
        bg = (toks[i], toks[i+1])
        if bg in seen: return 1.0
        seen.add(bg)
    return 0.0


# Encode images once per batch per CLIP tower
@torch.no_grad()
def clip_image_features(model: CLIPModel, proc: CLIPProcessor, images: List[Image.Image], device: str,
                        autocast_dtype=None) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = proc(images=images, return_tensors="pt")
    pix = inputs["pixel_values"]

    use_amp = (device.type == 'cuda' and dtype == torch.float16)
    ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else contextlib.nullcontext()
    
    with ctx:
        feats = model.get_image_features(pixel_values=pix)

    return feats.float()


# Encode texts (greedy / sampled) for a batch
@torch.no_grad()
def clip_text_features(model: CLIPModel, proc: CLIPProcessor, texts: List[str], device: str,
                       autocast_dtype=None) -> torch.Tensor:
    dev = next(model.parameters()).device
    toks = proc(text=texts, return_tensors="pt", padding=True)
    ids  = toks["input_ids"].to(dev)          # longs, don't cast
    attn = toks["attention_mask"].to(dev)
    # (model dtype handled internally; inputs stay integer/bool)
    feats = model.get_text_features(input_ids=ids, attention_mask=attn)
    
    return feats.float()


# ---------------------------
# Training loop (SCST)
# ---------------------------

def train(args):
    seed_everything(args.seed)
    device = args.device if args.device != "auto" else device_auto()
    print(f"Device: {device}")

    # --- Load models ---
    cap = BlipForConditionalGeneration.from_pretrained(args.blip_id)
    proc_blip = BlipProcessor.from_pretrained(args.blip_id)

    cap.to(device)
    cap.train()

    # Gradient checkpointing can save RAM on 2070
    if args.grad_ckpt:
        try:
            cap.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing.")
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

    # Optional LoRA on BLIP (target text decoder attn projections)
    if args.lora_r > 0:
        if not _HAS_PEFT:
            raise RuntimeError("LoRA requested but 'peft' is not installed. pip install peft")
        # Option A: hardcode BLIP-1 targets (BERT-style attention)
        
        target_modules = ["query", "key", "value"]

        # Option B: auto-detect targets (uncomment to use)
        # def guess_lora_targets(model):
        #     names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        #     bertish = [t for t in ["query","key","value"] if any(t in n for n in names)]
        #     if bertish: return bertish
        #     projish = [t for t in ["q_proj","k_proj","v_proj","out_proj"] if any(t in n for n in names)]
        #     if projish: return projish
        #     return ["query","value"]
        # target_modules = guess_lora_targets(cap)

        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_2_SEQ_LM",  # BLIP captioner is encoder-decoder with generate()
        )

        # If you’re also using 8-bit Adam, prep the base model first
        if args.adam8bit:
            from peft import prepare_model_for_kbit_training
            cap = prepare_model_for_kbit_training(cap)

            cap = get_peft_model(cap, lconf)
            cap.print_trainable_parameters()

            # (Optional) sanity: show a few adapted modules
            show = [n for n, _ in cap.named_modules() if "lora" in n][:6]
            print(f"[LoRA] attached to: {show}")
        else:
            print_trainable_params(cap, "BLIP")
            
        def load_clip_or_die(model_id, device, want_dim):
            cfg = AutoConfig.from_pretrained(model_id)
            assert cfg.model_type == "clip"
            assert cfg.text_config.hidden_size == want_dim and cfg.text_config.max_position_embeddings == 77
            m = CLIPModel.from_pretrained(model_id)
            p = CLIPProcessor.from_pretrained(model_id)
            return m.eval().to(device), p

        vitl, pvitl = load_clip_or_die(args.vitl_id, device="cuda", want_dim=768)

        if args.bigg_device == "cpu":
            bigg, pbigg = load_clip_or_die(args.bigg_id, device="cpu",  want_dim=1280)
        elif args.bigg_device == "cuda":
            # try half precision to fit
            bigg = CLIPModel.from_pretrained(args.bigg_id, torch_dtype=torch.float16).eval().to(args.bigg_device or 'cuda')
            pbigg = CLIPProcessor.from_pretrained(args.bigg_id)
        elif args.bigg_device == "8bit":
            # needs accelerate + bitsandbytes
            bigg = CLIPModel.from_pretrained(args.bigg_id, load_in_8bit=True, device_map={"": 0}).eval()
            pbigg = CLIPProcessor.from_pretrained(args.bigg_id)
            
    
    # a) Nuke the specific cached model dir (safe & targeted)
    #   mac/linux: rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14
    #   (Adjust path if you’re using HF_HOME)

    # b) Force a fresh download for this call
#    vitl = CLIPModel.from_pretrained(
 #       "openai/clip-vit-large-patch14",
#        local_files_only=False,
#        force_download=True, use_safetensors=True   # grab a clean copy
#    ).eval().to(device)

    # ...and make sure the processor matches the same ID:
#    pvitl = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=False, force_download=True, use_safetensors=True)

    # CLIP towers aligned with SDXL families
    vitl_id = args.vitl_id   # OpenAI ViT-L/14
    bigg_id = args.bigg_id   # OpenCLIP ViT-bigG/14

    vitl  = CLIPModel.from_pretrained(vitl_id).eval().to(device)
    pvitl = CLIPProcessor.from_pretrained(vitl_id)
    bigg  = CLIPModel.from_pretrained(bigg_id).eval().to(args.bigg_device)
    pbigg = CLIPProcessor.from_pretrained(bigg_id)
    
    cfg = AutoConfig.from_pretrained(vitl_id)
    print("text_hidden_size:", cfg.text_config.hidden_size,
        "vision_hidden_size:", cfg.vision_config.hidden_size,
        "max_pos:", cfg.text_config.max_position_embeddings)
    # Expect for ViT-L/14: hidden_size=768, max_pos=77

    for m in (vitl, bigg):
        for p in m.parameters(): p.requires_grad = False

    # Datasets & loader
    ds = ImageFolderNoLabels(args.data_dir)
    
    def collate_pils(batch):
    # batch is a list[PIL.Image.Image]; return as-is
        return batch

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_pils,     # <-- add this
        pin_memory=(device=="cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Optimizer
    if args.adam8bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not installed; remove --adam8bit or install it.")
        optimizer = bnb.optim.AdamW8bit(cap.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(cap.parameters(), lr=args.lr)

    # AMP (fp16) on CUDA
    use_fp16 = (device == "cuda") and args.fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    clip_autocast_dtype = torch.float16 if use_fp16 else None

    os.makedirs(args.out_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch_images in dl:
            # 0) Pre-cache CLIP image features for greedy & sampled (no grad)
            with torch.no_grad():
                img_vitl = clip_image_features(vitl, pvitl, batch_images, device, clip_autocast_dtype)
                img_bigg = clip_image_features(bigg, pbigg, batch_images, device, clip_autocast_dtype)
                img_vitl = F.normalize(img_vitl, dim=-1)
                img_bigg = F.normalize(img_bigg, dim=-1)

            # 1) Greedy baseline captions (no grad)
            inputs = proc_blip(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                out_g = cap.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=proc_blip.tokenizer.eos_token_id
                )
            greedy_txt = proc_blip.batch_decode(out_g, skip_special_tokens=True)

            # 2) Sampled captions (with scores) — these drive RL
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=proc_blip.tokenizer.eos_token_id
            )
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out_s = cap.generate(**inputs, **gen_kwargs)
            else:
                out_s = cap.generate(**inputs, **gen_kwargs)

            sampled_txt = proc_blip.batch_decode(out_s.sequences, skip_special_tokens=True)
            logp_sum = get_token_logp_sum(out_s.scores, out_s.sequences).to(device)  # [B]
            seq_len = out_s.sequences.size(1)

            # 3) Rewards: CLIP cosine with SDXL-aligned towers (text side changes per caption)
            with torch.no_grad():
                # cache image feats
                img_vitl = F.normalize(clip_image_features(vitl, pvitl, batch_images), dim=-1)  # on vitl's device/dtype
                img_bigg = F.normalize(clip_image_features(bigg, pbigg, batch_images), dim=-1)  # on bigg's device/dtype

                # text feats
                txt_vitl_s = F.normalize(clip_text_features(vitl, pvitl, sampled_txt), dim=-1)
                txt_bigg_s = F.normalize(clip_text_features(bigg, pbigg, sampled_txt), dim=-1)
                txt_vitl_b = F.normalize(clip_text_features(vitl, pvitl, greedy_txt),  dim=-1)
                txt_bigg_b = F.normalize(clip_text_features(bigg, pbigg, greedy_txt),  dim=-1)

		# cosines; bring bigG results onto vitl’s device before mixing
                main_dev = next(vitl.parameters()).device
                R_vitl_s = (img_vitl * txt_vitl_s).sum(-1)                             # on main_dev
                R_bigg_s = (img_bigg * txt_bigg_s).sum(-1).to(main_dev)                # move to main_dev
                R_vitl_b = (img_vitl * txt_vitl_b).sum(-1)
                R_bigg_b = (img_bigg * txt_bigg_b).sum(-1).to(main_dev)

                R_s = args.alpha_bigg * R_bigg_s + args.alpha_vitl * R_vitl_s
                R_b = args.alpha_bigg * R_bigg_b + args.alpha_vitl * R_vitl_b

            # 4) Anti-gaming penalties
            rep_flags = torch.tensor([has_repeat_bigram(t) for t in sampled_txt],
                                     device=device, dtype=torch.float32)
            reward = R_s - R_b - args.len_penalty*seq_len - args.rep_penalty*rep_flags  # [B]

            # 5) SCST loss
            loss = -(reward.detach() * logp_sum).mean()  # scalar
            
            # inside the loop, after computing R_s, R_b, sampled_txt, greedy_txt, seq_len
            if args.log_prompts and (step % args.log_every == 0):
                k = min(args.log_n, len(sampled_txt))
                color_on = not args.no_color
                dR = (R_s - R_b).tolist()
                Rs = R_s.tolist(); Rb = R_b.tolist()
                print(_c("\n--- Prompts (greedy vs sampled) ---", "y", color_on))
                for i in range(k):
                    g = _trunc(greedy_txt[i], args.truncate)
                    s = _trunc(sampled_txt[i], args.truncate)
                    delta = dR[i]
                    sig = _c(f"+{delta:.3f}", "g", color_on) if delta >= 0 else _c(f"{delta:.3f}", "r", color_on)
                    print(f"[{i}] " + _c("greedy:", "dim", color_on), g)
                    print(f"    " + _c("sample:", "b", color_on), s)
                    print(f"    " + _c("scores:", "dim", color_on),
                        f"R_b={Rb[i]:.3f}  R_s={Rs[i]:.3f}  ΔR={sig}")
                print(_c("--- end prompts ---\n", "y", color_on))

            # 6) Optim step (with grad accumulation + AMP)
            loss_to_accum = loss / max(1, args.grad_accum)
            if use_fp16:
                scaler.scale(loss_to_accum).backward()
            else:
                loss_to_accum.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(cap.parameters(), 1.0)
                if use_fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} "
                      f"| loss {loss.item():.4f} "
                      f"| R_s {R_s.mean().item():.3f} "
                      f"| R_b {R_b.mean().item():.3f} "
                      f"| ΔR {(R_s-R_b).mean().item():.3f}")

        # Save each epoch
        cap.save_pretrained(args.out_dir)
        proc_blip.save_pretrained(args.out_dir)
        print(f"Saved fine-tuned BLIP to {args.out_dir}")

    print("Done.")


# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="RL fine-tuning of BLIP-1 with CLIPScore reward aligned to SDXL (ViT-L/14 + bigG/14)."
    )
    # Data / output
    ap.add_argument("--data-dir", required=True, help="Folder of training images (recurses).")
    ap.add_argument("--out-dir",  required=True, help="Where to save fine-tuned BLIP weights.")
    # Model IDs (can be HF IDs or local dirs)
    ap.add_argument("--blip-id",  default="Salesforce/blip-image-captioning-large",
                    help="BLIP-1 captioning model or local dir.")
    ap.add_argument("--vitl-id",  default="openai/clip-vit-large-patch14",
                    help="OpenAI CLIP ViT-L/14 model or local dir.")
    ap.add_argument("--bigg-id",  default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                    help="OpenCLIP ViT-bigG/14 model or local dir.")
    # Reward mix
    ap.add_argument("--alpha-bigg", type=float, default=0.7, help="Weight for bigG tower reward.")
    ap.add_argument("--alpha-vitl", type=float, default=0.3, help="Weight for ViT-L tower reward.")
    # Decode
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--top-p",          type=float, default=0.9)
    ap.add_argument("--temperature",    type=float, default=1.0)
    # Penalties
    ap.add_argument("--len-penalty", type=float, default=0.05, help="Linear penalty per generated token.")
    ap.add_argument("--rep-penalty", type=float, default=0.10, help="Penalty if any bigram repeats.")
    # Train
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2, help="2 is a safe default for 8GB + bigG.")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad-accum", type=int, default=4, help="Accumulate steps to simulate larger batch.")
    ap.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing on BLIP.")
    ap.add_argument("--log-every", type=int, default=10)
    # Memory/perf knobs
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 AMP on CUDA (recommended on RTX 2070).")
    ap.add_argument("--adam8bit", action="store_true", help="Use bitsandbytes AdamW8bit (saves VRAM).")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = off).")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # System
    ap.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    # in build_argparser()
    ap.add_argument("--log-prompts", action="store_true",
                    help="Print greedy/sample captions & rewards at log interval.")
    ap.add_argument("--log-n", type=int, default=2,
                    help="How many items from each batch to print.")
    ap.add_argument("--truncate", type=int, default=140,
                    help="Truncate printed captions to this many chars.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    ap.add_argument("--bigg-device", default="cpu", choices=["cpu","cuda","8bit"],
                help="Where to keep the bigG CLIP model. 'cpu' avoids OOM on 8GB GPUs.")
    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
