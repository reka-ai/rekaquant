# Reka Quant
[Reka Quant](https://link_to_blogpost) is a model quantization library. It supports:
 - NF4 and  GGML (llama.cpp) quantization primitives.  GGML primitives are added directly from its source code through python cffi bindings, making it easy to incorporate new ones.
 - Exporting of GGML quantized models to native GGUF format, for easy integration with the existing ecosystem.
 - Activation-aware quantization by leveraging precomputed activation statistics from a text sample, through the LDLQ method from [QuIP](https://arxiv.org/pdf/2307.13304).
 - Further eror reduction through self-distillation from the BF16 model, while quantizing the network gradually.
 - Fast multi-node training through full or hybrid FSDP, as well as fast parallel proxy Hessian computation for LDLQ.
 
# Installation

Clone the library with submodules:

`git clone --recurse-submodules git@github.com:reka-ai/quantization.git`

Install requirements:

`poetry install`

Build the shared library in csrc, needed for python bindings. 
```
cd csrc
gcc -shared -o quantize.so -fPIC quantize.c
cd ..
```

Exporting to GGUF formats requires a patch to the llama.cpp library, apply it and install the library.
```
cd third_party/llama.cpp
git apply ../../patches/RekaQuant.patch
cmake -B build
cmake --build build --config Release
cd ../..
```


# Usage


The main script is `train.py`. The training data should be in jsonl format with documents in the "text" field.
```
torchrun \
    ...distributed flags.. \
    python3 src/train.py \
    --model_path $model_path \
    --ref_model $ref_model \
    --out_path $out_path \
    --train_data $train_data \
    --hessian_corr 1e-1 \
    --hessian_train_seq 4096   \
    --total_train_steps 1800 \
    --lr 1e-5 \
    --global_batch_size 512  \
    --seq_len 8192 \
    --micro_batch_size 1 \
    --checkpoint_iters 100 \
    --valid_seq 64 \
    --quant_strategy typewise_Q3_K_S \
    --use_checkpointing \
```

An example slurm script can be found in [run_train.slurm](run_train.slurm):
```
export REF_MODEL_PATH=/path/to/model
export OUT_PATH=/path/to/output
export TRAIN_DATA=/path/to/train.jsonl

sbatch run_train.slurm
```
When training smaller models, you can enable the --use_hybrid flag to use hybrid FSDP (shard intra-node, replicate across nodes) for reduced communication and higher efficiency, and remove the --use_checkpointing flag to disable activation checkpointing.

Once the model is trained, if you used GGUF quants you will need to export it to a native GGUF file. You can see the [scripts/prepare_ckpt.sh](scripts/prepare_ckpt.sh) script for an example of how to do this.
```
cd scripts
bash prepare_ckpt.sh $OUT_PATH/iter_001800/ #GGUF ckpt saved under $OUT_PATH/iter_001800/hf_model/Q3_K_S_RekaQuant_hf
```

**NOTE**: GGML K-Quants require tensors to have a number of columns divisible by 256. You can use the helper script in [scripts/pad_intermediate.py](scripts/pad_intermediate.py) if needed to preprocess models.
