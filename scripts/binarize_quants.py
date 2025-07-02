
import click
import torch
import os
import sys
import numpy as np
import re
import struct
sys.path.append(os.getenv('QUANT_DIR'))

from utils.ggml import dequantize_tensor


def permute(weights: np.ndarray, n_head: int, n_head_kv: int | None):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))

def _map_names(hf_name: str):
    assert hf_name.endswith('.weight'), f"Only weight tensors are supported: {hf_name}"
    hf_name = hf_name.removesuffix('.weight')

    direct_map = {
        "model.embed_tokens": "token_embd.weight",
        "lm_head": "output.weight",
    }
    if hf_name in direct_map:
        return direct_map[hf_name]

    layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", hf_name)
    if layer_match:
        layer_num = layer_match.group(1)
        component_part = layer_match.group(2)

        assert component_part != "input_layernorm", f"Input layer norm not supported: {hf_name}"
        assert component_part != "post_attention_layernorm", f"Post attention layer norm not supported: {hf_name}"
        assert component_part != "post_mlp_layernorm", f"Post mlp layer norm not supported: {hf_name}"

        # Attention and MLP blocks (split component and sub-component)
        parts = component_part.split('.')
        if len(parts) != 2:
            raise ValueError(f"Unexpected layer format: {hf_name}")

        component_hf = parts[0]
        sub_component_hf = parts[1]

        if component_hf == "self_attn":
            component_cpp = "attn"
            sub_map = {
                "q_proj": "q", "k_proj": "k",
                "v_proj": "v", "o_proj": "output"
            }
            if sub_component_hf in sub_map:
                return f"blk.{layer_num}.{component_cpp}_{sub_map[sub_component_hf]}.weight"
        elif component_hf == "mlp":
            component_cpp = "ffn"
            sub_map = {
                "gate_proj": "gate", "up_proj": "up", "down_proj": "down"
            }
            if sub_component_hf in sub_map:
                return f"blk.{layer_num}.{component_cpp}_{sub_map[sub_component_hf]}.weight"
            
    raise ValueError(f"Unrecognized tensor name format: {hf_name}")

@click.command()
@click.option('--quant_state_dir', required=True, type=str, help='Directory containing the quantization state.')
@click.option('--target_hf_dir', required=False, type=str, help='Directory containing the target HF model, if passed it will write the dequantized tensors to this target HF model.')
@click.option('--n_kv_heads', type=int, help='Number of KV heads.')
@click.option('--n_heads', type=int, help='Number of heads.')
def main(quant_state_dir: str, target_hf_dir: str | None, n_kv_heads: int, n_heads: int):
    dequant_tensors = {}
    tensor2size = {}
    for file in os.listdir(quant_state_dir):
        if file.endswith('.pt'): 
            print(f'Binarizing {file}')
            name = file.strip('.pt').replace('._fsdp_wrapped_module', '').replace('_fsdp_wrapped_module.', '').replace('._checkpoint_wrapped_module', '')+ '.weight'
            quant_state = torch.load(os.path.join(quant_state_dir, file), weights_only=False)

            quant_data = None

            if 'primitive' in quant_state:
                # No LDLQ
                primitive = quant_state['primitive']
                print(f'No LDLQ for {name}')
                dtype = quant_state['primitive']
                quant_data = quant_state['quant_state']
                metadata = quant_state['metadata']

                shape = metadata[0][0], metadata[0][1]
                bitwidth = quant_data.itemsize * 8 * quant_data.size / shape[0] / shape[1]
                print(f'Primitive: {primitive}, bitwidth: {bitwidth}')

                if target_hf_dir is not None:
                    out_features, in_features = shape
                    tensor = torch.zeros(out_features, in_features, dtype=torch.bfloat16)
                    dequant = dequantize_tensor(quant_data, ((out_features, in_features), out_features * in_features), dtype)
                    tensor[:, :] = torch.tensor(dequant)
                    dequant_tensors[name] = tensor

            else:
                # LDLQ
                keys = list(quant_state.keys())
                key_begins = [b for b,e in keys]
                key_ends = [e for b,e in keys]
                
                print(f'LDLQ for {name}, with {len(keys)} keys')
                assert min(key_begins) == 0 
                
                in_features = max(key_ends)

                assert in_features // 256 == len(keys), f'Expected a block size of 256 for LDLQ, but {in_features} // 256 != {len(keys)} (#keys)'

                metadata = quant_state[keys[0]]['metadata']
                dtype = quant_state[keys[0]]['primitive']
                
                out_features, in_features = metadata[0]
                shape = out_features, in_features

                # Read the quantized data pieces
                quant_data_pieces = []
                for key in sorted(keys):
                    try:
                        quant_data = quant_state[key]['quant_state']
                        quant_data_pieces.append(quant_data)
                        assert len(quant_data.shape) == 1
                    except Exception as e:
                        print(f'Error dequantizing {key}: {e}')
                        continue
                assert all(piece.size == quant_data_pieces[0].size for piece in quant_data_pieces), f'All pieces must have the same size'
                print(f'Quant data pieces shape: {quant_data_pieces[0].shape}')
                quant_piece_numel = quant_data_pieces[0].size
                block_quant_cols = quant_data_pieces[0].reshape(out_features, -1).shape[1]
                bitwidth = block_quant_cols * quant_data_pieces[0].itemsize * 8 / 256 # Calculate how many bits per original tensor element, since a block is 256 elements
                print(f'Quantized datatype {dtype} block has {block_quant_cols} columns, bitwidth of {bitwidth}')
                quant_total_numel = quant_piece_numel * len(quant_data_pieces)
                
                # Now stitch blocks of columns together
                quant_data = np.zeros(quant_total_numel, dtype=quant_data_pieces[0].dtype).reshape(out_features, -1)
                print(f'quant data shape: {quant_data.shape}, emplacing pieces of shape {quant_data_pieces[0].reshape(-1, block_quant_cols).shape}')
                for i, piece in enumerate(quant_data_pieces):
                    piece = piece.reshape(-1, block_quant_cols)
                    quant_data[:, i*block_quant_cols:(i+1)*block_quant_cols] = piece
                

                # Flatten quant data
                quant_data = quant_data.reshape(-1).copy(order='C')
                if target_hf_dir is not None:
                    tensor = torch.zeros(out_features, in_features, dtype=torch.bfloat16)
                    dequant = dequantize_tensor(quant_data, ((out_features, in_features), out_features * in_features), dtype)
                    tensor[:, :] = torch.tensor(dequant)
                    dequant_tensors[name] = tensor
                
                assert len(quant_data.shape) == 1
            




            quant_tensor = quant_data.reshape(shape[0], -1)            
            print(f'For {name},  shape: {shape} | quant data shape: {quant_data.shape} | quant tensor shape: {quant_tensor.shape}')

            #Permute to match llama.cpp format
            if name.endswith(("q_proj.weight")):
                quant_tensor = permute(quant_tensor, n_heads, n_heads)
                quant_data = quant_tensor.reshape(-1)
            if name.endswith(("k_proj.weight")):
                quant_tensor = permute(quant_tensor, n_heads, n_kv_heads)
                quant_data = quant_tensor.reshape(-1)


            assert len(quant_data.shape) == 1
            assert quant_data.dtype == np.uint8
            # Now save to C format

            hf_name = name.removesuffix('.pt')
            llama_cpp_name = _map_names(hf_name)
            out_path = os.path.join(quant_state_dir, f'{llama_cpp_name}.bin')
            quant_data = quant_data.copy(order='C')
            raw_bytes = quant_data.tobytes()
            num_bytes = len(raw_bytes) 


            # Save size header and raw bytes
            try:
                with open(out_path, 'wb') as f:
                    size_header = struct.pack('<Q', num_bytes)
                    f.write(size_header)
                    f.write(raw_bytes)
                    print(f"Wrote data ({len(raw_bytes)} bytes).")

                print(f"Successfully saved data to {out_path}")

            except IOError as e:
                print(f"Error writing to file {out_path}: {e}")
                exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during saving: {e}")
                exit(1)
            
            tensor2size[name] = quant_data.itemsize * quant_data.size
            size_so_far = sum([tensor2size[k] for k in tensor2size])
            print(f'Size so far: {size_so_far/1024/1024} MB')

    print(f'Total size: {sum([tensor2size[k] for k in tensor2size])/1024/1024} MB')
    if target_hf_dir is not None:
        print(f'Writing to target HF model at {target_hf_dir}')

        for tensor_file in os.listdir(target_hf_dir):
            if tensor_file.endswith('.bin'):
                print(f"Processing {tensor_file}")
                tensors = torch.load(os.path.join(target_hf_dir, tensor_file), map_location='cpu')
                for name, tensor in tensors.items():
                    if name in dequant_tensors:
                        print(f'{name} found in dequant_tensors')
                        diff_tensor = tensor - dequant_tensors[name]
                        print(f'Average abs diff: {diff_tensor.abs().mean()}')
                        print(f'Relative abs diff: {(diff_tensor/(tensor.abs() + 1e-6)).abs().mean()}')
                        print(f'Relative max diff: {(diff_tensor/(tensor.abs() + 1e-6)).abs().max()}')
                        tensor.data = dequant_tensors[name].to(tensor.device, dtype=tensor.dtype)
                    else:
                        print(f'{name} not found in dequant_tensors')
                torch.save(tensors, os.path.join(target_hf_dir, tensor_file))



if __name__ == '__main__':
    main()