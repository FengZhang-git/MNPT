# 1. Training
The training script is in scripts/mnpt/train.sh.
e.g., 16-shot training with ViT-B/16:

```CUDA_VISIBLE_DEVICES=0 bash scripts/mnpt/train.sh data imagenet vit_b16_ep30 16 300 10 positive_prompt_checkpoint 0.2```

# 2. Inference

The inference script is in scripts/mnpt/eval.sh.
e.g., 16-shot evaluation with ViT-B/16:
```CUDA_VISIBLE_DEVICES=0 bash scripts/mnpt/eval.sh data imagenet vit_b16_ep30 output/imagenet_MNPT/vit_b16_ep30_16shots_anneal0.2_10_np300_nctnegx16_miu10_t1/seed1 positive_prompt_checkpoint 300```
