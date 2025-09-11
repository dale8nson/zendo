import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from transformers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from pickle import dumps
from safetensors import serialize
from safetensors.torch import load_file, save_file
from safetensors.torch import save as st_save
import argparse
import os


class CUDATrainer:
    def __init__(
        self,
        model_path: str,
        data_file:str,
        lr: float=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, gradient_accumulation_steps=1, lr_warmup_steps=20, max_train_steps=2000, batch_size=1, num_batches=1, image_size=1024, checkpoint=0):

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

        data = load_file(data_file)
        token_embed   = data["0"]
        token_embed_2 = data["1"]            
        pids1    = data["2"]            
        pids2    = data["3"]
        am1 = data["4"]
        am2 = data["5"]       
        image_embeds  = data["8"]            
        timesteps     = data["9"]            
        targets       = data["10"]           
        time_ids      = data["11"]

        self.num_images = image_embeds.shape[0]

        self.enc = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder')
        self.enc2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder='text_encoder_2')

        for p in self.enc.parameters(): p.requires_grad_(False)
        for p in self.enc2.parameters(): p.requires_grad_(False)



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

        self.enc.text_model.encoder.requires_grad_(False)
        self.enc.text_model.final_layer_norm.requires_grad_(False)
        self.enc.text_model.embeddings.position_embedding.requires_grad_(False)
        self.enc2.text_model.encoder.requires_grad_(False)
        self.enc2.text_model.final_layer_norm.requires_grad_(False)
        self.enc2.text_model.embeddings.position_embedding.requires_grad_(False)

        self.enc.gradient_checkpointing_enable()
        self.enc2.gradient_checkpointing_enable()

        if is_xformers_available():
            import xformers
            self.unet.enable_xformers_memory_efficient_attention()

        self.emb = self.enc.get_input_embeddings()
        self.emb2 = self.enc2.get_input_embeddings()
        orig_vocab_1 = self.emb.num_embeddings
        orig_vocab_2 = self.emb2.num_embeddings
        self.token_id   = orig_vocab_1
        self.token_id_2 = orig_vocab_2
        need1 = orig_vocab_1 + 1
        need2 = orig_vocab_2 + 1

        self.enc.resize_token_embeddings(need1)
        self.enc2.resize_token_embeddings(need2)
        
        self.emb = self.enc.get_input_embeddings()
        self.emb2 = self.enc2.get_input_embeddings()
        
        
        self.checkpoint = checkpoint
        
        if checkpoint > 0:
            state = load_file(f'ti_step_{checkpoint}.safetensors')  # safetensors has no map_location
            token_embed   = state['embedding_1'].to(device)
            token_embed_2 = state['embedding_2'].to(device)
            print(f"Loaded checkpoint_{checkpoint}.safetensors")
        
        self.emb.weight.data[self.token_id]   = token_embed.to(dtype=self.emb.weight.dtype, device=self.emb.weight.device)
        self.emb2.weight.data[self.token_id_2] = token_embed_2.to(dtype=self.emb2.weight.dtype, device=self.emb2.weight.device)
        
        self.index_no_updates = torch.ones(self.emb.num_embeddings, dtype=torch.bool, device=self.emb.weight.device)
        self.index_no_updates[self.token_id] = False

        self.index_no_updates_2 = torch.ones(self.emb2.num_embeddings, dtype=torch.bool, device=self.emb2.weight.device)
        self.index_no_updates_2[self.token_id_2] = False
        
        self.orig_embeds_params   = self.emb.weight.detach().clone()
        self.orig_embeds_params_2 = self.emb2.weight.detach().clone()
        
        self.emb.requires_grad_(True)
        self.emb2.requires_grad_(True)
        
        def _check_placeholder_coverage(self):
            c1 = 0; c2 = 0; nseq = 0
            for b in self.samples:
                p1 = b["pids1"]; p2 = b["pids2"]
                c1 += (p1 == self.token_id  ).sum().item()
                c2 += (p2 == self.token_id_2).sum().item()
                nseq += p1.numel() // p1.shape[-1]  # number of sequences
            print(f"[data check] placeholder token count — enc1:{c1} enc2:{c2} over {nseq} seqs")
        
        _check_placeholder_coverage(self)
        
        self.lr = (
            lr * gradient_accumulation_steps * self.batch_size * self.accelerator.num_processes
        )

        self.optimizer = AdamW(
            [
                {"params": [self.emb.weight],  "lr": lr},  # enc1: a little faster
                {"params": [self.emb2.weight], "lr": lr},  # enc2: slowed down
            ],
            betas=betas, eps=eps, weight_decay=weight_decay
            )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps

        num_warmup_steps_for_scheduler = lr_warmup_steps * self.accelerator.num_processes

        num_training_steps_for_scheduler = self.max_train_steps * self.accelerator.num_processes

        self.lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

        self.enc.train()
        self.enc2.train()

        assert self.enc.get_input_embeddings().num_embeddings  == need1
        assert self.enc2.get_input_embeddings().num_embeddings == need2
        assert self.token_id   == orig_vocab_1
        assert self.token_id_2 == orig_vocab_2
        
        # ---- per-encoder LR scales (won’t be clobbered by scheduler) ----
        self.lr_scale_1 = 1.00      # enc1 starts at 1.00×
        self.lr_scale_2 = 1.20      # enc2 starts +20% (often needs more early)
        self.lr_scale_min = 0.25
        self.lr_scale_max = 4.00

        # remember your initial group order: [enc1, enc2]
        base_p1 = self.optimizer.param_groups[0]
        base_p2 = self.optimizer.param_groups[1]
        self.base_lr_1 = base_p1["lr"]
        self.base_lr_2 = base_p2["lr"]


        self.enc, self.enc2, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.enc,
            self.enc2,
            self.optimizer,
            self.lr_scheduler
        )

    def train(self, global_step: int = 0, first_epoch: int = 0, epochs: int = 1, threshold=2.5, cooldown=400):
        batch_size = self.batch_size
        max_train_steps = self.max_train_steps
        
        def mask_all_but_first(ids, am, token_id):
            # ids: [B, L], am: [B, L] (1=attend)
            is_tok = (ids == token_id)
            csum = is_tok.int().cumsum(dim=1)
            dup = is_tok & (csum > 1)
            am = am.clone()
            am[dup] = 0  # ignore duplicate occurrences
            return am
        
        def clip_placeholder_grads_(embedding, token_id: int, max_norm: float = 150.0):
            w = embedding.weight
            g = w.grad
            if g is None:
                return

            row = g[token_id]  # view into grad

            # ---- sanitize NaNs/Infs (choose ONE of the two blocks below) ----
            # A) Preferred (in-place, no 'out='):
            try:
                row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            except TypeError:
                # Older PyTorch fallback (no posinf/neginf or kw mismatch):
                row.copy_(torch.where(torch.isfinite(row), row, torch.zeros_like(row)))

            # ---- clip the grad row norm ----
            rn = row.norm().item()
            if rn > max_norm:
                row.mul_(max_norm / (rn + 1e-12))

        def empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
        global_step = self.checkpoint if self.checkpoint > 0 else global_step

        initial_global_step = global_step

        remain  = self.max_train_steps - global_step
        warmup  = max(1, int(0.03 * remain))
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup, num_training_steps=remain
        )
        
        
        cos_band_low      = 0.24     # a bit lower so it's reachable sooner
        cooldown_factor   = 0.5      # halve LR in polish
        cooldown_steps    = cooldown      # slightly shorter polish window
        hard_cap          = 0.55
        # min_steps_total   = max(1500, global_step + 500)  # don’t stop before this
        # allow_polish_after = global_step + 300       # don’t polish before this     
        self.lr_scale_1 = 0.85      # was 1.00 — slow enc1 a bit from the start
        self.lr_scale_2 = 1.40      # was 1.20 — speed enc2 a bit from the start
        self.lr_scale_min = 0.25
        self.lr_scale_max = 4.00      

        entered_band   = False
        cooldown_left  = 0
        
        total_batch_size = batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps

        print("***** Running training *****")
        print(f"  Num examples = {self.num_batches}")
        print(f"  Num Epochs = {epochs}")
        print(f"  Instantaneous batch size per device = {batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")

        progress_bar = tqdm(range(0, max_train_steps), initial=initial_global_step, desc="Steps")

        empty_cache()
        
        try:
            for epoch in range(first_epoch, epochs):
                self.enc.train()
                self.enc2.train()

                accum = self.gradient_accumulation_steps
                stop = False 
                    
                i = 0
                while global_step < max_train_steps:
                    if stop:
                        break
                    batch = self.samples[i % len(self.samples)]
                    i += 1


                    with self.accelerator.accumulate([self.enc, self.enc2]):
                        # ---- device + dtypes ----
                        lat  = batch["latents"].to(self.device, dtype=torch.float16)       # [B,4,H,W] fp16
                        tgt  = batch["target"].to(self.device, dtype=torch.float16)        # [B,4,H,W] fp16
                        t    = batch["t"].to(self.device, dtype=torch.int64)
                        am1 = batch["am1"].to(self.device)
                        am2 = batch["am2"].to(self.device)
                        pids1   = batch["pids1"].to(self.device)           # [B,L1,768] fp32
                        pids2   = batch["pids2"].to(self.device)
                        tids = batch["time_ids"].to(self.device, dtype=torch.int64)        # [B,6]
                        
                        am1 = mask_all_but_first(pids1, am1, self.token_id)
                        am2 = mask_all_but_first(pids2, am2, self.token_id_2)

                        enc_output = self.enc(pids1.long(), attention_mask=am1.long(), output_hidden_states=True)
                        enc2_output = self.enc2(pids2.long(), attention_mask=am2.long(),  output_hidden_states=True)

                        hs1 = enc_output.hidden_states[-2].to(dtype=torch.float16)
                        hs2 = enc2_output.hidden_states[-2].to(dtype=torch.float16)
                        text_embeds = enc2_output[0].to(dtype=torch.float16)

                        added = {"text_embeds": text_embeds, "time_ids": tids}

                        hs = torch.cat([hs1, hs2], dim=-1)
                        
                        # if (pids1 == self.token_id).any() or (pids2 == self.token_id_2).any(): continue

                        # ---- UNet forward ----
                        pred = self.unet(lat.to(self.device), t, hs.to(self.device), added_cond_kwargs=added).sample    # [B,4,H,W]

                        # mse = torch.nn.functional.mse_loss(pred.float(), tgt.float(), reduction="none")
                        # print(f"t mean={t.float().mean():.1f}  mse mean={mse.mean().item():.4f}")

                        # ---- loss / backward (no AMP scaler) ----
                        loss = F.mse_loss(pred.float(), tgt.float(), reduction="mean") / accum

                        self.accelerator.backward(loss)

                        # ----- PRE-STEP (freeze + mask) -----
                        if self.accelerator.sync_gradients:
                            # drift before the update (used for freeze choice; no prints here)
                            d1, d2 = self.cos()
                            
                            w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
                            w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight
                            cur1, cur2 = w1[self.token_id], w2[self.token_id_2]
                            base1,base2 = self.orig_embeds_params[self.token_id], self.orig_embeds_params_2[self.token_id_2]

                            cos = torch.nn.functional.cosine_similarity
                            d1 = 1 - cos(cur1.to(self.device), base1.to(self.device), dim=0).item()
                            d2 = 1 - cos(cur2.to(self.device), base2.to(self.device), dim=0).item()
                            
                            # optionally freeze the fast side's placeholder row for a few steps
                            # if hasattr(self, "_freeze_ticks") and self._freeze_ticks > 0:
                            #     if d1_pre > d2_pre:
                            #         if self.emb.weight.grad is not None:
                            #             self.emb.weight.grad[self.token_id].zero_()
                            #     else:
                            #         if self.emb2.weight.grad is not None:
                            #             self.emb2.weight.grad[self.token_id_2].zero_()
                            #     self._freeze_ticks -= 1

                            # zero grads for ALL non-placeholder rows
                            if self.emb.weight.grad is not None:
                                self.emb.weight.grad[self.index_no_updates] = 0
                            if self.emb2.weight.grad is not None:
                                self.emb2.weight.grad[self.index_no_updates_2] = 0

                            # <<< MOVE THESE OUT OF THE emb2 if-block (always safe; helper early-returns) >>>
                            # clip_placeholder_grads_(self.emb,  self.token_id,   max_norm=(100.0 if not entered_band else 80.0))
                            # clip_placeholder_grads_(self.emb2, self.token_id_2, max_norm=(100.0 if not entered_band else 80.0))

                            # (optional) grad probe every 50 steps (pre-update)
                            if (global_step % 50) == 0:
                                w1 = self.emb.weight
                                w2 = self.emb2.weight
                                g1 = 0.0 if w1.grad is None else float(w1.grad[self.token_id].norm())
                                g2 = 0.0 if w2.grad is None else float(w2.grad[self.token_id_2].norm())
                                print(f"[grad probe] ||∂L/∂ti1||={g1:.3e}  ||∂L/∂ti2||={g2:.3e}")

                            # ----- optimizer step + scheduler -----
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            sched_lr = self.lr_scheduler.get_last_lr()[0]

                            # reapply per-encoder LR scales
                            
                            d1, d2 = self.cos()
                            
                            upper = max(d1, d2)
                            lower = min(d1, d2)
                            
                            w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
                            w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight
                            
                            p1 = self.optimizer.param_groups[0]
                            p2 = self.optimizer.param_groups[1]
                            
                            if d1 < d2:
                                
                                
                            # calculate the new lr based on the lr  required to achieve the same cosine similarity of the faster moving encoder
                            # the new weights are calculated by subtracting the product of the lr, gradient and the loss from the current weights
                            # the cosine similarity score is the dot product of the normalised unit vector of the current and original token embeddings
                            # 
                            
                            
                            #     # new_lr = (d2 / grad1 / loss)[0].item()
                                p1['lr'] *= 2
                                p2['lr'] *= 0.25
                                
                            elif d2 < d1:
                            #     #  new_lr = (d1 / grad2 / loss)[0].item()
                                p2['lr'] *= 2
                                p1['lr'] *= 0.25
                           
                                 
                            # p1 = self.optimizer.param_groups[0]
                            # p2 = self.optimizer.param_groups[1]
                            # p1["lr"] = float(sched_lr * self.lr_scale_1)
                            # p2["lr"] = float(sched_lr * self.lr_scale_2)

                            self.optimizer.zero_grad(set_to_none=True)

                            # renormalize placeholder rows to their original norms
                            with torch.no_grad():
                                w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
                                w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight
                                for (w, tid, base) in (
                                    (w1, self.token_id,   self.orig_embeds_params),
                                    (w2, self.token_id_2, self.orig_embeds_params_2),
                                ):
                                    e  = w[tid]
                                    n0 = base[tid].norm()
                                    e.mul_(n0 / e.norm().clamp_min(1e-6))

                            global_step += 1
                            progress_bar.update(1)
                            self.accelerator.log(
                                {"loss": loss.detach().item() * self.gradient_accumulation_steps, "lr": sched_lr},
                                step=global_step,
                            )

                            # save a checkpoint every 500 steps
                            if (global_step % 500) == 0:
                                with torch.no_grad():
                                    w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
                                    w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight
                                    e1 = w1[self.token_id  ].detach().to(device="cpu", dtype=torch.float16).contiguous()
                                    e2 = w2[self.token_id_2].detach().to(device="cpu", dtype=torch.float16).contiguous()
                                    save_file({"embedding_1": e1, "embedding_2": e2}, f"ti_step_{global_step}.safetensors")

                            # ----- POST-STEP (telemetry + LR balance + polish) -----
                            with torch.no_grad():
                                d1, d2 = self.cos()

                                # print drift periodically
                                if (global_step % 25) == 0:
                                    print(f"Δcos enc1:{d1:.8f} enc2:{d2:.8f}")
                                    print(f'enc lr: {p1["lr"]:.5e}  enc2 lr: {p2["lr"]:.5e}')

                                # enter polish only when BOTH encoders are “in band” and aligned
                                both_over = (d1 >= cos_band_low) and (d2 >= cos_band_low)
                                if (not entered_band) and (both_over):
                                    entered_band  = True
                                    cooldown_left = cooldown_steps

                                    # one-time alignment nudge at entry: push scales toward the slower side
                                    align_gain = 0.20
                                    gap_now = float(d1 - d2)      # >0 => enc1 ahead
                                    delta   = max(-0.20, min(0.20, align_gain * gap_now))
                                    self.lr_scale_1 *= (1.0 - delta)
                                    self.lr_scale_2 *= (1.0 + delta)

                                    # halve both for polish "cooldown"
                                    self.lr_scale_1 *= cooldown_factor
                                    self.lr_scale_2 *= cooldown_factor

                                    print(f"Entered polish at step {global_step}. "
                                        f"scales: {self.lr_scale_1:.2f}/{self.lr_scale_2:.2f} "
                                        f"(d1={d1:.5f}, d2={d2:.5f})")

                                # polish stop conditions
                                if entered_band:
                                    cooldown_left -= 1
                                    stop_now = (
                                        (d1 > hard_cap or d2 > hard_cap) or
                                        (cooldown_left <= 0)
                                    )
                                    if stop_now:
                                        print(f"Stopping at step {global_step}: Δcos enc1:{d1:.3f} enc2:{d2:.3f}")
                                        stop = True

                    with torch.no_grad():
                        self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight[self.index_no_updates]   = \
                            self.orig_embeds_params [self.index_no_updates].to(self.device)
                        self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight[self.index_no_updates_2] = \
                            self.orig_embeds_params_2[self.index_no_updates_2].to(self.device)
                           

                        empty_cache()

                self.accelerator.wait_for_everyone()
                
        except Exception as e:
            from traceback import print_tb
            print(f'{print_tb(e.__traceback__)}')
            print(f"Exception: {e}")
            

        except KeyboardInterrupt:
            pass

        with torch.no_grad():
            w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
            w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight

            e1 = w1[self.token_id  ].detach().to(device="cpu", dtype=torch.float16).contiguous()
            e2 = w2[self.token_id_2].detach().to(device="cpu", dtype=torch.float16).contiguous()

            save_file({"embedding_1": e1, "embedding_2": e2}, "output.safetensors")
            save_file({"embedding_1": e1, "embedding_2": e2}, f"ti_step_{global_step}.safetensors")

    def cos(self):
        w1 = self.accelerator.unwrap_model(self.enc ).get_input_embeddings().weight
        w2 = self.accelerator.unwrap_model(self.enc2).get_input_embeddings().weight
        cur1, cur2 = w1[self.token_id], w2[self.token_id_2]
        base1,base2 = self.orig_embeds_params[self.token_id], self.orig_embeds_params_2[self.token_id_2]

        cos = torch.nn.functional.cosine_similarity
        d1 = 1 - cos(cur1.to(self.device), base1.to(self.device), dim=0).item()
        d2 = 1 - cos(cur2.to(self.device), base2.to(self.device), dim=0).item()
        
        return d1,d2

        #     payload: bytes = st_save({"embedding_1": embedding_1, "embedding_2": embedding_2})

        # return payload


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
