#!/usr/bin/env python3
# BLIPit.py
import os, sys, argparse, glob, random, warnings, logging
from typing import List
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import contextlib

# HF Transformers
from transformers import (
    BlipForConditionalGeneration, BlipProcessor,
    CLIPModel, CLIPProcessor, AutoConfig
)
from transformers.utils import logging as hf_logging

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

# Work around TP plan verification bug in some Transformers builds
try:  # pragma: no cover - defensive
    from transformers.integrations import tensor_parallel as _tp
    from transformers import modeling_utils as _mu

    def _squelch_verify(func):
        def _wrapper(expected_keys, plan):
            try:
                return func(expected_keys, plan)
            except ValueError:
                return None
        _wrapper.__name__ = f"_blipit_{func.__name__}"
        return _wrapper

    if hasattr(_tp, "verify_tp_plan") and not getattr(_tp.verify_tp_plan, "__name__", "").startswith("_blipit"):
        _tp.verify_tp_plan = _squelch_verify(_tp.verify_tp_plan)
    if hasattr(_mu, "verify_tp_plan") and not getattr(_mu.verify_tp_plan, "__name__", "").startswith("_blipit"):
        _mu.verify_tp_plan = _squelch_verify(_mu.verify_tp_plan)
except Exception:
    pass


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
    sequences : [B, T+1], includes BOS at position 0
    returns   : [B] sum of log-probs of sampled tokens
    """
    seq = sequences[:, 1:]  # drop BOS
    logps = []
    for t, scores_t in enumerate(gen_scores):
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
def clip_image_features(model: CLIPModel, proc: CLIPProcessor, images: List[Image.Image], *_, **__) -> torch.Tensor:
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = proc(images=images, return_tensors="pt")
    pix = inputs["pixel_values"].to(dev)
    if dev.type == "cuda" and dtype == torch.float16:
        pix = pix.to(torch.float16)
    else:
        pix = pix.to(torch.float32)
    use_amp = (dev.type == "cuda" and dtype == torch.float16)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
    with ctx:
        feats = model.get_image_features(pixel_values=pix)
    return feats.float()

# Encode texts for a batch
@torch.no_grad()
def clip_text_features(model: CLIPModel, proc: CLIPProcessor, texts: List[str], device: str,
                       autocast_dtype=None) -> torch.Tensor:
    dev = next(model.parameters()).device
    toks = proc(text=texts, return_tensors="pt", padding=True)
    ids  = toks["input_ids"].to(dev)
    attn = toks["attention_mask"].to(dev)
    feats = model.get_text_features(input_ids=ids, attention_mask=attn)
    return feats.float()

def load_clip_or_die(model_id, want_dim):
    cfg = AutoConfig.from_pretrained(model_id)
    assert cfg.model_type == "clip", f"{model_id} is {cfg.model_type}, need 'clip'"
    assert cfg.text_config.hidden_size == want_dim and cfg.text_config.max_position_embeddings == 77, \
        f"{model_id}: hidden={cfg.text_config.hidden_size}, max_pos={cfg.text_config.max_position_embeddings}"
    m = CLIPModel.from_pretrained(model_id).eval()   # do NOT .to() here
    p = CLIPProcessor.from_pretrained(model_id)
    return m, p


def _sum_logprobs_for_sequences(cap: BlipForConditionalGeneration,
                                pixel_values: torch.Tensor,
                                sequences: torch.Tensor,
                                use_amp: bool = False) -> torch.Tensor:
    """
    Returns per-example sum of log probabilities for the given decoded sequences
    under the current model parameters (teacher forcing on the sampled tokens).

    sequences: [B, T] token ids as returned by generate (includes BOS at idx 0)
    """
    device = next(cap.parameters()).device
    seq = sequences.to(device)
    # Shift for teacher forcing: predict seq[:, 1:] from inputs seq[:, :-1]
    if seq.size(1) <= 1:
        return torch.zeros(seq.size(0), device=device, dtype=torch.float32)

    dec_in = seq[:, :-1].contiguous()
    labels = seq[:, 1:].contiguous()

    pad_id = cap.generation_config.pad_token_id
    if pad_id is None:
        pad_id = cap.config.pad_token_id
    if pad_id is None:
        pad_id = cap.config.eos_token_id
    if pad_id is None:
        pad_id = 0

    labels_mask = (labels != pad_id)
    labels_for_gather = labels.masked_fill(~labels_mask, 0)

    # Forward
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
    with ctx:
        attn = dec_in.ne(pad_id).long()
        out = cap(pixel_values=pixel_values, input_ids=dec_in, attention_mask=attn)
        logits = out.logits  # [B, T-1, V]
        logp = F.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, labels_for_gather.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        gathered = gathered * labels_mask.float()
        per_example = gathered.sum(dim=1)
    return per_example


# ---------------------------
# Training loop (SCST)
# ---------------------------

def train(args):
    seed_everything(args.seed)
    device = args.device if args.device != "auto" else device_auto()
    print(f"Device: {device}")

    warnings.filterwarnings("ignore", message=r".*use_cache=True.*gradient checkpointing.*")
    for name in (
        "transformers.modeling_utils",
        "transformers.generation.utils",
        "transformers.models.blip.modeling_blip_text",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    if args.quiet_hf:
        hf_logging.set_verbosity_error()

    # --- Load models ---
    cap = BlipForConditionalGeneration.from_pretrained(args.blip_id)
    proc_blip = BlipProcessor.from_pretrained(args.blip_id)

    # Silence the "use_cache=True incompatible with gradient checkpointing" spam
    cap.config.use_cache = False

    cap.to(device)
    cap.train()

    # Gradient checkpointing can save RAM on small GPUs
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
        target_modules = ["query", "key", "value"]
        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        if args.adam8bit:
            from peft import prepare_model_for_kbit_training
            cap = prepare_model_for_kbit_training(cap)
        else:
            print_trainable_params(cap, "BLIP")
        cap = get_peft_model(cap, lconf)
        cap.print_trainable_parameters()
        show = [n for n, _ in cap.named_modules() if "lora" in n][:6]
        print(f"[LoRA] attached to: {show}")

    # --- CLIP towers (load once) ---
    vitl, pvitl = load_clip_or_die(args.vitl_id, 768)
    bigg, pbigg = load_clip_or_die(args.bigg_id, 1280)

    # Place towers on devices (ViT-L on main device; bigG default CPU to avoid 8GB OOM)
    vitl = vitl.to(device)
    if args.bigg_device == "cuda":
        try:
            bigg = bigg.to("cuda")
            print("bigG on CUDA")
        except torch.cuda.OutOfMemoryError:
            print("bigG OOM on CUDA; falling back to CPU")
            bigg = bigg.to("cpu")
    elif args.bigg_device == "8bit":
        raise NotImplementedError("Wire up 8-bit only if accelerate/bitsandbytes are installed.")
    else:
        bigg = bigg.to("cpu")

    print("vitl device:", next(vitl.parameters()).device)
    print("bigg device:", next(bigg.parameters()).device)

    for m in (vitl, bigg):
        for p in m.parameters():
            p.requires_grad = False

    # Inspect ViT-L config (sanity)
    cfg = AutoConfig.from_pretrained(args.vitl_id)
    print("text_hidden_size:", cfg.text_config.hidden_size,
          "vision_patch:", getattr(cfg.vision_config, 'patch_size', 'n/a'))

    # --- Data ---
    ds = ImageFolderNoLabels(args.data_dir)
    def _collate(batch):
        return batch  # list[Image]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=_collate, drop_last=True)

    # --- Optimizer ---
    params = [p for p in cap.parameters() if p.requires_grad]
    if args.adam8bit and _HAS_BNB:
        opt = bnb.optim.AdamW8bit(params, lr=args.lr)
        print("Using AdamW8bit")
    else:
        opt = torch.optim.AdamW(params, lr=args.lr)
        print("Using AdamW")

    scaler = torch.amp.GradScaler('cuda', enabled=(args.fp16 and device == 'cuda'))

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Train ---
    global_step = 0
    for epoch in range(args.epochs):
        for it, images in enumerate(dl):
            cap.train()

            # Prepare BLIP inputs
            blip_inputs = proc_blip(images=images, return_tensors="pt")
            pix = blip_inputs["pixel_values"].to(device)

            # Generate greedy (baseline)
            gen_kwargs = dict(
                pixel_values=pix,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
            if args.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            with torch.no_grad():
                seq_greedy = cap.generate(**gen_kwargs)

            # Generate sampled
            gen_kwargs_s = dict(
                pixel_values=pix,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                use_cache=False,
            )
            if args.no_repeat_ngram_size > 0:
                gen_kwargs_s["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            with torch.no_grad():
                seq_sampled = cap.generate(**gen_kwargs_s)

            # Decode to text
            texts_g = proc_blip.tokenizer.batch_decode(seq_greedy, skip_special_tokens=True)
            texts_s = proc_blip.tokenizer.batch_decode(seq_sampled, skip_special_tokens=True)

            tgt = torch.device(device)

            # CLIP image features (once per tower)
            img_vitl = clip_image_features(vitl, pvitl, images).to(tgt)
            img_bigg = clip_image_features(bigg, pbigg, images).to(tgt)

            # CLIP text features for greedy/sampled
            txt_g_vitl = clip_text_features(vitl, pvitl, texts_g, device).to(tgt)
            txt_g_bigg = clip_text_features(bigg, pbigg, texts_g, device).to(tgt)
            txt_s_vitl = clip_text_features(vitl, pvitl, texts_s, device).to(tgt)
            txt_s_bigg = clip_text_features(bigg, pbigg, texts_s, device).to(tgt)

            # Rewards (cosine sims)
            r_g = args.alpha_vitl * normalize_cosine(img_vitl, txt_g_vitl) + \
                  args.alpha_bigg * normalize_cosine(img_bigg, txt_g_bigg)
            r_s = args.alpha_vitl * normalize_cosine(img_vitl, txt_s_vitl) + \
                  args.alpha_bigg * normalize_cosine(img_bigg, txt_s_bigg)

            # Penalties: length (per generated token) and repeated bigram
            pad_id = proc_blip.tokenizer.pad_token_id or 0
            len_g = (seq_greedy.ne(pad_id).sum(dim=1) - 1).clamp_min(0).float()
            len_s = (seq_sampled.ne(pad_id).sum(dim=1) - 1).clamp_min(0).float()
            if args.len_penalty != 0.0:
                r_g = r_g - args.len_penalty * len_g
                r_s = r_s - args.len_penalty * len_s
            if args.rep_penalty != 0.0:
                rep_g = torch.tensor([has_repeat_bigram(t) for t in texts_g], device=r_g.device)
                rep_s = torch.tensor([has_repeat_bigram(t) for t in texts_s], device=r_s.device)
                r_g = r_g - args.rep_penalty * rep_g
                r_s = r_s - args.rep_penalty * rep_s

            # Advantage
            adv = r_s - r_g
            if args.normalize_reward:
                mu = adv.mean()
                sigma = adv.std().clamp_min(1e-6)
                adv = (adv - mu) / sigma
            if args.clip_floor is not None:
                gate = torch.sigmoid((r_s - args.clip_floor) / max(args.tau_soft, 1e-6))
                adv = adv * gate
            if args.adv_margin > 0.0:
                adv = torch.where(adv.abs() < args.adv_margin, torch.zeros_like(adv), adv)
            if args.adv_clip is not None and args.adv_clip > 0:
                adv = adv.clamp(-args.adv_clip, args.adv_clip)
            if args.adv_scale != 1.0:
                adv = adv * args.adv_scale

            # Policy gradient loss via teacher-forced log-probs of sampled tokens
            use_amp = (args.fp16 and device == 'cuda')
            logp_sum = _sum_logprobs_for_sequences(cap, pix, seq_sampled, use_amp=use_amp)
            rl_loss = -(adv.detach() * logp_sum).mean()

            # Backward / step with grad accumulation
            loss = rl_loss / max(1, args.grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = ((it + 1) % args.grad_accum == 0)
            if do_step:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            # Logging
            if (global_step % args.log_every) == 0:
                avg_r_g = r_g.mean().item()
                avg_r_s = r_s.mean().item()
                print(f"ep{epoch} it{it} step{global_step} | Rg={avg_r_g:.4f} Rs={avg_r_s:.4f} Adv={(avg_r_s-avg_r_g):.4f} Loss={rl_loss.item():.4f}")
                if args.log_prompts:
                    k = min(args.log_n, len(images))
                    print("-- Samples --")
                    for j in range(k):
                        tg = _trunc(texts_g[j], args.truncate)
                        ts = _trunc(texts_s[j], args.truncate)
                        print(f"[{j}] G: {tg}")
                        print(f"    S: {ts}")
                        print(f"    Rg={r_g[j].item():.4f} Rs={r_s[j].item():.4f} Adv={(r_s[j]-r_g[j]).item():.4f}")

            global_step += 1

        # Save checkpoint each epoch
        try:
            if hasattr(cap, 'save_pretrained'):
                cap.save_pretrained(args.out_dir)
            if hasattr(proc_blip, 'save_pretrained'):
                proc_blip.save_pretrained(args.out_dir)
            print(f"Saved to {args.out_dir}")
        except Exception as e:
            print(f"Warning: failed to save checkpoint: {e}")


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
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 AMP on CUDA.")
    ap.add_argument("--adam8bit", action="store_true", help="Use bitsandbytes AdamW8bit (saves VRAM).")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = off).")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # System
    ap.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    ap.add_argument("--bigg-device", default="cpu", choices=["cpu","cuda","8bit"],
                    help="Where to keep the bigG CLIP model. 'cpu' avoids OOM on 8GB GPUs.")
    ap.add_argument("--log-prompts", action="store_true",
                    help="Print greedy/chosen captions & rewards at log interval.")
    ap.add_argument("--log-n", type=int, default=2,
                    help="How many items from each batch to print.")
    ap.add_argument("--truncate", type=int, default=140,
                    help="Truncate printed captions to this many chars.")
    # Advantage shaping
    ap.add_argument("--adv-margin", type=float, default=0.0,
                    help="Dead-band around zero advantage (ignore tiny changes).")
    ap.add_argument("--adv-clip", type=float, default=5.0,
                    help="Clamp advantage to [-adv_clip, adv_clip].")
    ap.add_argument("--adv-scale", type=float, default=1.0,
                    help="Multiply the advantage by this factor.")
    ap.add_argument("--normalize-reward", action="store_true",
                    help="Normalize advantage by batch mean/std before margin.")
    ap.add_argument("--clip-floor", type=float, default=None,
                    help="If set, prefer samples with reward >= this value.")
    ap.add_argument("--tau-soft", type=float, default=0.02,
                    help="Softness for threshold gate (larger = smoother).")
    # Logging control
    ap.add_argument("--quiet-hf", action="store_true",
                    help="Silence HuggingFace warnings (including use_cache/ckpt).")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0,
                help="Block repeating n-grams of this size during generation (0 disables).")
    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
