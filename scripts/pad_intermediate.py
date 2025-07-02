import os
import json
import math
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import argparse

def get_next_multiple_of_256(value):
    return math.ceil(value / 256) * 256

def check_and_pad_tensor(tensor, old_intermediate_size, new_intermediate_size, tensor_name):
    modified = False
    tensor_shape = list(tensor.shape)
    
    for dim_idx, dim_size in enumerate(tensor_shape):
        if dim_idx > 0:  # Skip first dimension for checking multiples
            if dim_size % 256 != 0:
                if dim_size != old_intermediate_size:
                    raise ValueError(f"Tensor {tensor_name} has dimension {dim_idx} with size {dim_size} "
                                   f"which is not a multiple of 256 and not equal to intermediate_size.")
                      
    for dim_idx, dim_size in enumerate(tensor_shape):
        if dim_size == old_intermediate_size:
            tensor_shape[dim_idx] = new_intermediate_size
            modified = True
    
    if not modified:
        return tensor
        
    new_tensor = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    
    indices = tuple(slice(0, min(tensor.shape[i], new_tensor.shape[i])) for i in range(len(tensor_shape)))
    
    new_tensor[indices] = tensor[indices]
    
    return new_tensor

def convert_checkpoint(model_path):
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    old_intermediate_size = config.get('intermediate_size')
    if old_intermediate_size is None:
        raise ValueError("intermediate_size not found in config.json")
    
    if old_intermediate_size % 256 == 0:
        print(f"intermediate_size ({old_intermediate_size}) is already a multiple of 256. No adjustment needed.")
        new_intermediate_size = old_intermediate_size
    else:
        new_intermediate_size = get_next_multiple_of_256(old_intermediate_size)
        print(f"Adjusting intermediate_size from {old_intermediate_size} to {new_intermediate_size}")
        
        config['intermediate_size'] = new_intermediate_size
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Updated config.json with new intermediate_size: {new_intermediate_size}")
    
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if not safetensors_files:
        print("No .safetensors files found in the model directory.")
        return
    
    for safetensor_file in safetensors_files:
        file_path = os.path.join(model_path, safetensor_file)
        print(f"Processing {safetensor_file}...")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            
            tensor_names = f.keys()
            tensors_dict = {}
            
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                
                padded_tensor = check_and_pad_tensor(tensor, old_intermediate_size, new_intermediate_size, tensor_name)
                tensors_dict[tensor_name] = padded_tensor
        
        save_file(tensors_dict, file_path, metadata={"format": "pt"})

        print(f"Saved updated tensors to {file_path}")
    
    print("Model conversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Convert Llama model checkpoint to ensure dimensions are multiples of 256")
    parser.add_argument("model_path", type=str, help="Path to the model directory containing config.json and safetensors files")
    args = parser.parse_args()
    
    try:
        convert_checkpoint(args.model_path)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()