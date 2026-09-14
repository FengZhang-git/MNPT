import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()

def neg_select_topk(p, neg, fullneg, top_k, label, num_of_local_feature, cfg):
    label_repeat = label.repeat_interleave(num_of_local_feature)

    pred_topk = torch.topk(p, k=top_k, dim=1)[1]

    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]
    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda(), torch.tensor([0]).cuda()

    selected_neg = neg[~contains_label]
    max_neg, pos_neg = torch.max(selected_neg, dim=1, keepdim=True)

    fullneg_trans = fullneg.transpose(0, 1)
    pos_neg = pos_neg.squeeze(dim=1)
    
    selected_fullneg = torch.stack([fullneg_trans[idx.item()] for idx in pos_neg])
    
    ood_logits = torch.concat((max_neg, selected_fullneg), dim=1)
    ood_labels = torch.zeros(ood_logits.shape[0], dtype=torch.long).cuda()
    loss_ood = F.cross_entropy(ood_logits, ood_labels)
    
    loss_diversity = F.cross_entropy(selected_neg, pos_neg)
    
    return loss_ood, loss_diversity

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:

        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, neg_prompts, tokenized_neg_prompts):
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        y = neg_prompts + self.positional_embedding.type(self.dtype)
        y = y.permute(1, 0, 2)
        y, _, _, _ = self.transformer(y)
        y = y.permute(1, 0, 2)
        y = self.ln_final(y).type(self.dtype)
        y = y[torch.arange(y.shape[0]), tokenized_neg_prompts.argmax()] @ self.text_projection

        return x, y

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        self.pos_ctx_path = cfg.TRAINER.MNPT.POSITIVE_WEIGHTS
        
        n_ctx_neg = cfg.TRAINER.MNPT.N_CTX_NEG
        n_ctx_np = cfg.TRAINER.MNPT.N_CTX_NP
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        neg_ctx_vectors = torch.empty(n_ctx_np, n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(neg_ctx_vectors, std=0.02)
        neg_prompt_prefix = " ".join(["X"] * n_ctx_neg)
        
        print(f'Initial negative context: "{neg_prompt_prefix}"')
        print(f"Number of negative context words: {n_ctx_neg}")
        print(f"Number of negative prompts: {n_ctx_np}")

        self.neg_ctx = nn.Parameter(neg_ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        checkpoint = None
        if osp.isfile(self.pos_ctx_path):
            print(f"Loading positive context from {self.pos_ctx_path}")
            checkpoint = load_checkpoint(self.pos_ctx_path)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            
            if "ctx" in checkpoint:
                ctx_shape = checkpoint["ctx"].shape
                if len(ctx_shape) == 3:
                    n_ctx = ctx_shape[1]
                else:
                    n_ctx = ctx_shape[0]
                prompt_prefix = " ".join(["X"] * n_ctx)
            else:
                print("Warning: could not find 'ctx' in checkpoint")

                n_ctx = cfg.TRAINER.MNPT.N_CTX_FALLBACK
                prompt_prefix = " ".join(["X"] * n_ctx)
        else:

            print(f"Warning: positive context path {self.pos_ctx_path} not found")
            print("Initializing with default values (these won't be trained)")
            n_ctx = cfg.TRAINER.MNPT.N_CTX_FALLBACK
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
        neg_tokenized_prompts = clip.tokenize(neg_prompt_prefix)
        with torch.no_grad():
            neg_embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        
        self.register_buffer("neg_token_prefix", neg_embedding[:, :1, :])
        self.register_buffer("neg_token_suffix", neg_embedding[:, 1 + n_ctx_neg:, :])
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_np = n_ctx_np
        self.tokenized_prompts = tokenized_prompts
        self.neg_tokenized_prompts = neg_tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MNPT.CLASS_TOKEN_POSITION
        
        if checkpoint is not None and "ctx" in checkpoint:
            print('Loading positive context from checkpoint')

            pos_ctx = checkpoint["ctx"]
            self.register_buffer("pos_ctx", pos_ctx)
        else:
            print('No positive context found in checkpoint, initializing with random values')

            if cfg.TRAINER.MNPT.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.register_buffer("pos_ctx", ctx_vectors)
    
    def forward(self):

        ctx = self.pos_ctx
        
        if ctx.dim() == 2:

            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([
                    prefix_i,
                    ctx_i_half1,
                    class_i,
                    ctx_i_half2,
                    suffix_i,
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([
                    prefix_i,
                    class_i,
                    ctx_i,
                    suffix_i,
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Invalid class_token_position: {self.class_token_position}")
        
        neg_ctx = self.neg_ctx
        neg_prefix = self.neg_token_prefix
        neg_suffix = self.neg_token_suffix
        
        neg_prompts = torch.cat([
            torch.cat([neg_prefix]*self.n_ctx_np),
            neg_ctx,
            torch.cat([neg_suffix]*self.n_ctx_np),
        ], dim=1)
        
        return prompts, neg_prompts

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.annealed_temperature = cfg.annealed_temperature
        self.min_temperature = cfg.min_temperature
        self.annealed_rate = cfg.annealed_rate
        self.init_temperature = cfg.init_temperature
        self.current_temp = nn.Parameter(torch.ones([]) * cfg.min_temperature, requires_grad=False)
    
    def set_temperature(self, epoch):
        if self.annealed_temperature:
            self.current_temp.data = torch.ones([]) * max(self.min_temperature, self.init_temperature*np.exp(-self.annealed_rate * epoch))
        return self.current_temp.item()

    def forward(self, image):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))
        
        prompts, neg_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        tokenized_neg_prompts = self.neg_tokenized_prompts
        
        text_features, neg_text_features = self.text_encoder(
            prompts, tokenized_prompts, neg_prompts, tokenized_neg_prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T
        
        if self.annealed_temperature:
            logits_neg = 1 / self.current_temp * image_features @ neg_text_features.t()
            logits_local_neg = 1 / self.current_temp * local_image_features @ neg_text_features.T
        elif self.min_temperature > 0:
            logits_neg = 1 / self.min_temperature * image_features @ neg_text_features.t()
            logits_local_neg = 1 / self.min_temperature * local_image_features @ neg_text_features.T
        else:
            logits_neg = logit_scale * image_features @ neg_text_features.t()
            logits_local_neg = logit_scale * local_image_features @ neg_text_features.T
        return logits, logits_local, logits_neg, logits_local_neg

@TRAINER_REGISTRY.register()
class MNPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MNPT.PREC in ["fp16", "fp32", "amp"]

        assert cfg.TRAINER.MNPT.POSITIVE_WEIGHTS, "POSITIVE_WEIGHTS must be specified for MNPT"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.miu_value = cfg.miu_value
        self.top_k = cfg.topk

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MNPT.PREC == "fp32" or cfg.TRAINER.MNPT.PREC == "amp":

            clip_model.float()

        print("Building custom CLIP model with pre-trained positive prompts")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "neg_ctx" not in name:
                param.requires_grad_(False)
            else:
                print(f"Training parameter: {name}")

        self.model.to(self.device)
        
        trainable_params = [p for p in self.model.prompt_learner.parameters() 
                           if p.requires_grad]
        
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MNPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MNPT.PREC

        current_temp = self.model.set_temperature(self.epoch)

        if prec == "amp":
            with autocast():
                output, output_local, output_neg, output_local_neg = self.model(image)
                
                batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                output_local = output_local.view(batch_size * num_of_local_feature, -1)
                output_local_neg = output_local_neg.view(batch_size * num_of_local_feature, -1)
                
                loss_ood, loss_diversity = neg_select_topk(output_local, output_local_neg, output_neg,
                                                          self.top_k, label, 
                                                          num_of_local_feature, self.cfg)
                
                loss = loss_ood + self.miu_value * loss_diversity

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, output_neg, output_local_neg = self.model(image)
            
            batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
            output_local = output_local.view(batch_size * num_of_local_feature, -1)
            output_local_neg = output_local_neg.view(batch_size * num_of_local_feature, -1)
            
            loss_ood, loss_diversity = neg_select_topk(output_local, output_local_neg, output_neg,
                                                      self.top_k, label, 
                                                      num_of_local_feature, self.cfg)
            
            loss = loss_ood + self.miu_value * loss_diversity

            self.model_backward_and_update(loss)

        acc = compute_accuracy(output, label)[0].item()
        
        loss_summary = {
            "loss": loss.item(),
            "loss_ood": loss_ood.item(),
            "loss_diversity": loss_diversity.item(),
            "acc": acc,
            "temp": current_temp,
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "neg_token_prefix" in state_dict:
                del state_dict["neg_token_prefix"]

            if "neg_token_suffix" in state_dict:
                del state_dict["neg_token_suffix"]

            if "pos_ctx" in state_dict:
                del state_dict["pos_ctx"]

            print("Loading weights to {} from '{}' (epoch = {})".format(name, model_path, epoch))

            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) > 1:
                output = output[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T, NEGT, args):
        def to_np(x): return x.data.cpu().numpy()
        def concat(x): return np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        alpha = args.alpha_value
        glmcm_score = []
        mcm_score = []
        mnpt_score = []
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.cuda()

            output, output_local, output_neg, output_local_neg = self.model_inference(images)
            
            output /= 100.0
            output_local /= 100.0
            
            smax_global = to_np(F.softmax(output/T, dim=-1))
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.append(mcm_global_score)
            glmcm_score.append(mcm_global_score+mcm_local_score)

            if self.cfg.annealed_temperature or self.cfg.min_temperature > 0:
                output_neg *= self.cfg.min_temperature
                output_local_neg *= self.cfg.min_temperature
            else:
                output_neg /= 100.0
                output_local_neg /= 100.0

            pos_softmax = np.max(to_np(F.softmax(output/T, dim=-1)), axis=1)
            neg_softmax = np.max(to_np(F.softmax(output_neg/NEGT, dim=-1)), axis=1)
            local_pos_softmax = np.max(to_np(F.softmax(output_local/T, dim=-1)), axis=(1, 2))
            local_neg_softmax = np.min(np.max(to_np(F.softmax(output_local_neg/NEGT, dim=-1)), axis=-1),axis=-1)
            mnpt_score = - (pos_softmax+local_pos_softmax) + alpha * (neg_softmax+local_neg_softmax)
            mnpt_score.append(mnpt_score)


        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), concat(mnpt_score)[:len(data_loader.dataset)].copy()

    @torch.no_grad()
    def test_visualize(self, img_path, label):
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local, output_neg, output_local_neg = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        pred_topk = torch.topk(output_local, k=200, dim=1)[1]

        contains_label = pred_topk.eq(torch.tensor(
            label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label
