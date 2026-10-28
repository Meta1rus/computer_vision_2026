---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-VL-3B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: lab9_adapter
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lab9_adapter

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on the lab9_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1148

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- training_steps: 100

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0847        | 0.625 | 20   | 0.1251          |
| 0.0757        | 1.25  | 40   | 0.1109          |
| 0.0674        | 1.875 | 60   | 0.1138          |
| 0.0999        | 2.5   | 80   | 0.1163          |
| 0.1111        | 3.125 | 100  | 0.1148          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.14.1
- Datasets 4.0.0
- Tokenizers 0.22.2