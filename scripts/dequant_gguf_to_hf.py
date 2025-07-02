import click 
import torch
from transformers import AutoModelForCausalLM 
import os 
@click.command()
@click.option("--gguf_file_path", type=str, help="Path to the input GGuf file")
def dequant_gguf_to_hf(gguf_file_path):
    #orig hf path is parent of the gguf file
    orig_hf_path = os.path.dirname(gguf_file_path)
    target_hf_path = gguf_file_path.replace(".gguf", "_hf/")

    model = AutoModelForCausalLM.from_pretrained(orig_hf_path, gguf_file=gguf_file_path, torch_dtype=torch.bfloat16)
    print(f'Model loaded from {orig_hf_path} and gguf file {gguf_file_path}')
    print(f'Saving model to {target_hf_path}')
    model.save_pretrained(target_hf_path)

if __name__ == "__main__":
    dequant_gguf_to_hf()