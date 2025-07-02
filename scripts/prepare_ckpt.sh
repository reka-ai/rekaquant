#!/bin/bash
#This script creates a GGUF quantized ckpt from a RekaQuant ckpt, ass well as dequantizing back to an HF model
set -x 

cd $(dirname $0)
CKPT_PATH=$1
KV_HEADS=$2
N_HEADS=$3
TOKENIZER_PATH=$4

PARENT_DIR=$(dirname "$CKPT_PATH")
if [ ! -d "$CKPT_PATH/hf_model" ]; then
    echo "Error: $CKPT_PATH/hf_model does not exist"
    exit 1
fi

cp $TOKENIZER_PATH/tokenizer*  $CKPT_PATH/hf_model/


echo "Binarizing quant states"

BASE_DIR=$(dirname $0)/..
BASE_DIR=$(realpath $BASE_DIR)

QUANT_DIR=$BASE_DIR/src python3 binarize_quants.py \
--quant_state_dir $PARENT_DIR/quant_states/ \
--n_kv_heads $KV_HEADS \
--n_heads $N_HEADS 

cd  ../third_party/llama.cpp 

python3 convert_hf_to_gguf.py \
 $CKPT_PATH/hf_model/ \
 --outfile $CKPT_PATH/hf_model/F16.gguf

./build/bin/llama-quantize \
--quant_states_path $PARENT_DIR/quant_states/ \
$CKPT_PATH/hf_model/F16.gguf \
$CKPT_PATH/hf_model/Q3_K_S_RekaQuant.gguf \
Q3_K_S 

cd ../../scripts


python3 dequant_gguf_to_hf.py \
--gguf_file_path $CKPT_PATH/hf_model/Q3_K_S_RekaQuant.gguf 

cp $TOKENIZER_PATH/tokenizer* $CKPT_PATH/hf_model/Q3_K_S_RekaQuant_hf/
