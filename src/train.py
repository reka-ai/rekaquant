import argparse
import functools
import glob
import json
import os
import socket
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Local imports
from utils.quantize import (
    quantize_LDLQ,
    quantize_q3_K,
    quantize_q4_K,
    quantize_q6_K,
)


class Tee:
    """Redirect stdout to both console and file."""
    
    def __init__(self, name: str, mode: str) -> None:
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    
    def __del__(self) -> None:
        sys.stdout = self.stdout
        self.file.close()
    
    def write(self, data: str) -> None:
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self) -> None:
        self.file.flush()


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--resume_iter', type=int, default=None, required=False)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--ref_model', type=str, required=True, default=None)
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--seq_len', type=int, default=4096)
parser.add_argument('--global_batch_size', type=int, default=1024)
parser.add_argument('--hessian_train_seq', type=int, default=2048)
parser.add_argument('--valid_seq', type=int, default=256)
parser.add_argument('--micro_batch_size', type=int, default=1)
parser.add_argument('--total_train_steps', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--hessian_corr', type=float, default=1e-2)
parser.add_argument('--disable_ldlq', action='store_true')
parser.add_argument('--use_hybrid', action='store_true')
parser.add_argument('--use_checkpointing', action='store_true')
parser.add_argument('--checkpoint_iters', type=int, default=0)
parser.add_argument('--quant_strategy', type=str, choices=['typewise_Q3_K_S', 'noquant', 'allatonce_Q3_K_S'], default='typewise_Q3_K_S')


# Distributed variables
RANK: int = 0
WORLD_SIZE: int = 1
LOCAL_WORLD_SIZE: int = 1
LOCAL_RANK: int = 0
LOCAL_GROUP: Optional[dist.ProcessGroup] = None
CROSS_GROUP: Optional[dist.ProcessGroup] = None


def pprint(*args: Any, allrank: bool = False) -> None:
    """Print with rank prefix."""
    global RANK
    if allrank or RANK == 0:
        print(f"RANK {RANK}:", *args, flush=True)


def init_dist() -> None:
    """Initialize distributed training."""
    global RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, LOCAL_GROUP, CROSS_GROUP
    RANK = int(os.environ['RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    LOCAL_WORLD_SIZE = int(os.environ['LOCAL_WORLD_SIZE'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(LOCAL_RANK)
    local_GPU = torch.cuda.current_device()
    pprint(f"RANK {RANK} LOCAL_RANK {LOCAL_RANK}  initialized hostname {socket.gethostname()} GPU {local_GPU}", allrank=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert rank == RANK
    torch.cuda.set_device(LOCAL_RANK)

    # Create local node group 
    N_NODES = WORLD_SIZE // LOCAL_WORLD_SIZE
    for i in range(N_NODES):
        node_ranks = list(range(i*LOCAL_WORLD_SIZE, (i+1)*LOCAL_WORLD_SIZE))
        dist.new_group(ranks=node_ranks, backend='nccl')
        if RANK in node_ranks:
            assert LOCAL_GROUP is None
            LOCAL_GROUP = dist.new_group(ranks=node_ranks)
    # Create cross node group
    for i in range(LOCAL_WORLD_SIZE):
        cross_ranks = list(range(i, WORLD_SIZE, LOCAL_WORLD_SIZE))
        assert len(cross_ranks) == N_NODES
        dist.new_group(ranks=cross_ranks, backend='nccl')
        if RANK in cross_ranks:
            assert CROSS_GROUP is None
            CROSS_GROUP = dist.new_group(ranks=cross_ranks)


def cleanup_dist() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_psd(mat: torch.Tensor) -> bool:
    """Check if matrix is positive semi-definite."""
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())


def dbg(t: torch.Tensor, name: str) -> None:
    """Debug tensor for non-finite values."""
    if not torch.isfinite(t).all():
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        finite = t[torch.isfinite(t)]
        min_, max_, mean_ = (
            finite.min().item() if finite.numel() else float('nan'),
            finite.max().item() if finite.numel() else float('nan'),
            finite.mean().item() if finite.numel() else float('nan'),
        )
        pprint(f"ERROR: {name}: nan={nan_cnt} inf={inf_cnt} "
              f"min={min_:.3e} max={max_:.3e} mean={mean_:.3e} total={t.numel()}", allrank=True)
        raise RuntimeError(f"Non-finite detected in {name} on rank {RANK}")


def register_H_hook(module: nn.Linear, scale: float) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, int]]:
    """Register hook to compute Hessian approximation."""
    assert scale > 0, f'Scale must be positive, got {scale}'
    global RANK, LOCAL_WORLD_SIZE
    n = module.in_features
    assert n % LOCAL_WORLD_SIZE == 0
    H = torch.zeros(n//LOCAL_WORLD_SIZE, n, dtype=torch.float64, device='cuda')
    mu = torch.zeros(n, dtype=torch.float64, device='cuda')
    ct = 0

    def H_hook(module: nn.Module, x_in: Tuple[torch.Tensor]) -> None:
        nonlocal H, mu, ct, n
        x : torch.Tensor = x_in[0].detach()
        x = x.reshape(-1, n)

        dbg(x, "x-in")
        mu_local = x.sum(dim=0).cuda()
        dist.all_reduce(mu_local, group=LOCAL_GROUP)
        mu.add_(mu_local)
        dbg(mu, "mu")
        full_H = torch.mm(x.T, x).cuda().to(torch.float64)/scale
        dbg(full_H, "full_h")

        full_H2 = full_H.clone()
        dist.all_reduce(full_H2, group=LOCAL_GROUP)
        dbg(full_H2, "full_H2_after_all_reduce")
        full_H2_slice = full_H2[LOCAL_RANK*n//LOCAL_WORLD_SIZE:(LOCAL_RANK+1)*n//LOCAL_WORLD_SIZE]

        H_local = torch.zeros(n//LOCAL_WORLD_SIZE, n, dtype=torch.float64, device='cuda')
        dbg(H_local, "H_local_after_reduce_scatter")
        dist.reduce_scatter_tensor(H_local, full_H, op=dist.ReduceOp.SUM, group=LOCAL_GROUP)

        diff = (H_local - full_H2_slice).flatten()

        assert torch.allclose(H_local, full_H2_slice, atol=1e-6, rtol=1e-4), f'Rank {RANK} H_local {H_local} full_H2_slice {full_H2_slice}, diff mean {torch.mean(torch.abs(H_local-full_H2_slice))} diff max {torch.max(torch.abs(H_local-full_H2_slice))}'
        H.add_(H_local)
        
        local_ct = torch.tensor(len(x), dtype=torch.float64).cuda()
        ct_orig = local_ct.clone()
        dist.all_reduce(local_ct, group=LOCAL_GROUP)
        assert torch.allclose(local_ct , ct_orig*LOCAL_WORLD_SIZE)
        ct += int(local_ct.item())

    hook = module.register_forward_pre_hook(H_hook)

    def done() -> Tuple[torch.Tensor, torch.Tensor, int]:
        nonlocal H, mu, ct, hook

        dist.all_reduce(mu, group=CROSS_GROUP)
        dist.all_reduce(H, group=CROSS_GROUP)
        local_ct = torch.tensor(ct, dtype=torch.float64).cuda()
        dist.all_reduce(local_ct, group=CROSS_GROUP)
        ct = int(local_ct.item())

        output_list = [torch.zeros_like(H) for _ in range(LOCAL_WORLD_SIZE)]
        dist.all_gather(output_list, H, group=LOCAL_GROUP)

        output_list = [Hl.cpu() for Hl in output_list]
        H = torch.cat(output_list, dim=0)

        mu.div_(ct)
        H.div_(ct).mul_(scale)
        mu = mu
        H = H.cpu()
        H = torch.triu(H) + torch.triu(H, diagonal=1).T 

        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def register_model_hooks(
    model: FSDP, 
    scale: float = 1.0, 
    hook_names: List[str] = []
) -> Dict[str, Callable[[], Tuple[torch.Tensor, torch.Tensor, int]]]:
    """Register hooks on model modules."""
    module_hooks = {}

    for name, module in model.named_modules():
        if (hasattr(module, '_fsdp_wrapped_module') and 
            isinstance(module._fsdp_wrapped_module, torch.nn.Linear) and 
            not any([k in name for k in ['k_proj', 'v_proj', 'gate_proj']]) and 
            'layers.' in name):
            if name in hook_names:
                pprint(f'Adding hook to {name}._fsdp_wrapped_module')
                done_hook = register_H_hook(module._fsdp_wrapped_module, scale)
                module_hooks[name] = done_hook
            else:
                pprint(f'Skipping hook to {name}._fsdp_wrapped_module because it is not in requested hook names')
    return module_hooks


def extract_hook_hessians(
    module_hooks: Dict[str, Callable[[], Tuple[torch.Tensor, torch.Tensor, int]]]
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
    """Extract Hessian approximations from hooks."""
    extracted_hooks = {}
    for name, hook in module_hooks.items():
        H, mu, ct = hook()
        H, mu, ct = H.cpu(), mu.cpu(), ct
        del hook

        torch.cuda.empty_cache()
        extracted_hooks[name] = (H, mu, ct)
    return extracted_hooks


def regularize_H(H: torch.Tensor, n: int, sigma_reg: float = 5e-2) -> torch.Tensor:
    """Regularize Hessian matrix."""
    H.div_(torch.diag(H).mean())
    idx = torch.arange(n)
    H[idx, idx] += sigma_reg
    return H


def get_quant_primitive(q_type: str) -> Any: #Quant primitive can return either only the quantized+dequantized tensor, or the quantized+dequantized tensor plus the quantized state
    """Get quantization primitive function."""
    primitive_map : Dict[str, Any] = {
        'q4_K': quantize_q4_K,
        'q3_K': quantize_q3_K,
        'q6_K': quantize_q6_K,
    }
    if q_type in primitive_map:
        return primitive_map[q_type]
    else:
        raise ValueError(f'Unknown quantization primitive {q_type}')


def save_model(model: FSDP, out_path: str, train_step: int, extra_args: Dict[str, Any]) -> None:
    """Save model checkpoint."""
    pprint(f'Saving model at iteration {train_step}')
    start_time = time.time()
    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
    with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
        save_path = os.path.join(out_path, f'iter_{train_step:06d}', 'hf_model')
        args_path = os.path.join(out_path, f'iter_{train_step:06d}', 'args.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(args_path, 'w') as f:
            json.dump(extra_args, f)
        
        full_state = model.state_dict()
        if RANK == 0:
            model.save_pretrained(save_path, safe_serialization=False, state_dict=full_state)
    dist.barrier()
    end_time = time.time()
    pprint(f"Time taken by save_model: {end_time - start_time} seconds")


def save_optimizer(optimizer: optim.Optimizer, out_path: str, train_step: int) -> None:
    """Save optimizer state."""
    pprint(f'Saving optimizer RANK {RANK} at iteration {train_step}', allrank=True)
    start_time = time.time()
    rank_optim_name = f'optimizer_{RANK:02d}.pt'
    save_path = os.path.join(out_path, f'iter_{train_step:06d}', rank_optim_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(optimizer.state_dict(), save_path)
    end_time = time.time()
    pprint(f"Time taken by save_optimizer: {end_time - start_time} seconds")


def grouped_modules_to_quantize(
    model: FSDP, 
    mode: str = 'typewise_Q3_K_S'
) -> List[List[Tuple[str,str]]]:
    """Group modules for quantization."""
    if mode == 'typewise_Q3_K_S':
        order = [
            [ ('up_proj', 'q3_K'), ], 
            [ ('gate_proj', 'q3_K'), ],
            [ ('down_proj', 'q3_K'), ],
            [ ('v_proj', 'q3_K'), ],
            [ ('o_proj', 'q3_K'), ],
            [ ('q_proj', 'q3_K'), ('k_proj', 'q3_K'), ],
            [ ('embed_tokens', 'q3_K'), ('lm_head', 'q6_K'), ],
        ]
    elif mode == 'allatonce_Q3_K_S':
        order = [
            [ ('up_proj', 'q3_K'), ('gate_proj', 'q3_K'), ('down_proj', 'q3_K'), ('v_proj', 'q3_K'), ('o_proj', 'q3_K'), ('q_proj', 'q3_K'), ('k_proj', 'q3_K'), ('embed_tokens', 'q3_K'), ('lm_head', 'q6_K'), ],
        ]
    elif mode == 'noquant':
        order = [
            [ ('empty_group', 'None') , ],
        ]
    else:
        raise ValueError(f'Unknown quantization strategy {mode}')
    
    grouped = []
    for types in order:
        group = []
        for name, module in model.named_modules():
            if (hasattr(module, '_fsdp_wrapped_module') and 
                isinstance(module._fsdp_wrapped_module, torch.nn.Linear) and 
                ( 'layers.' in name or 'lm_head' in name or 'embed_tokens' in name)):
                for t, q_type in types:
                    if t in name:
                        group.append((name, q_type))
            elif (hasattr(module, '_fsdp_wrapped_module') and 
                  isinstance(module._fsdp_wrapped_module, torch.nn.Embedding) and 
                  ('embed_tokens' in name)): 
                for t, q_type in types:
                    if t in name:
                        group.append((name, q_type))

        grouped.append(group)
    return grouped


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    datasets.disable_caching()
    n_repeats = 0

    with open(os.path.join(args.out_path, 'args.jsonl'), 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f)

    log_path = os.path.join(args.out_path, f'train.log')
    tee = Tee(log_path, 'a')

    # Set rank
    global RANK
    global WORLD_SIZE
    global LOCAL_WORLD_SIZE
    global LOCAL_RANK
    global LOCAL_GROUP
    global CROSS_GROUP

    cache_name = args.train_data + f'_seq_{args.seq_len}.cache'
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.allowed_special_tokens = {'<|endoftext|>', '<|endofprompt|>', '<|fim_middle|>', '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_eot_id|>', '<|fim_pad_id|>', '<|fim_eot|>' }
    
    if os.path.exists(cache_name):
        pprint(f'Loading cached dataset from {cache_name}')
        dataset = datasets.load_from_disk(cache_name)
    else:
        # Build only on rank 0
        pprint(f'Building dataset from {args.train_data} on RANK 0')
        if RANK == 0:
            dataset = load_dataset('json', data_files=args.train_data, split='train', cache_dir="~/hftmp")
            dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
            pprint(dataset)

            def tokenize(element: Dict[str,  Any]) -> Dict[str, torch.Tensor]:
                for i in range(len(element["text"])):
                    element["text"][i] = element["text"][i].rstrip()
                    if not element["text"][i].endswith("<|endoftext|>"):
                        element["text"][i] += "<|endoftext|>"
                    special_tokens = { "<|endofprompt|>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_eot_id|>", "<|fim_pad_id|>", "<|fim_eot|>"}
                    if any([t in element["text"][i] for t in special_tokens]):
                        pprint(f'WARNING: Found special token in text: {element["text"][i]}')
                try:
                    outputs = tokenizer(
                        element["text"],
                    )
                except Exception as e:
                    pprint(f'ERROR: Error tokenizing text: {element["text"]}: {e}')
                    raise e

                input_ids = outputs["input_ids"]
                input_ids = [item for sublist in input_ids for item in sublist]
                input_ids = torch.tensor(input_ids, dtype=torch.long)

                if len(input_ids) % args.seq_len != 0:
                    removed = (len(input_ids) % args.seq_len)
                    removed_percent = removed / len(input_ids) 
                    pprint(f'WARNING: Removing {removed} tokens ({removed_percent*100:.2f}%) from the end of the dataset batch to make it divisible by {args.seq_len}')
                    input_ids = input_ids[:-(len(input_ids) % args.seq_len)]

                final_outputs : Dict[str, torch.Tensor] = {
                    "input_ids": input_ids.reshape(-1, args.seq_len)
                }

                return final_outputs
            
            dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names, num_proc=50)
            pprint(f'Caching dataset to {cache_name}')
            dataset.save_to_disk(cache_name)
        dist.barrier()
        pprint(f'Waiting for build dataset on all processes', allrank=True)
        dataset = datasets.load_from_disk(cache_name)

    pprint(dataset)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    train_sampler : DistributedSampler = DistributedSampler(dataset["train"])
    eval_sampler : DistributedSampler = DistributedSampler(dataset["test"])

    train_dataloader = DataLoader(
        dataset["train"], shuffle=False, batch_size=args.micro_batch_size, collate_fn=data_collator, sampler=train_sampler, drop_last=True
    )
    eval_dataloader = DataLoader(
        dataset["test"], shuffle=False, batch_size=args.micro_batch_size, collate_fn=data_collator, drop_last=True, sampler=eval_sampler
    )

    pprint(f'Loaded {len(train_dataloader)} train batches and {len(eval_dataloader)} eval batches')

    granular_wrap_policy = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: isinstance(module, LlamaDecoderLayer) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding),
    )
 
    resume_step = 0

    if args.resume_iter is not None:
        pprint(f'Loading model from iteration {args.resume_iter}')
        iter_path = os.path.join(args.model_path, f'iter_{args.resume_iter:06d}', 'hf_model')
        assert os.path.exists(iter_path), f'Iteration path {iter_path} does not exist'
        model = LlamaForCausalLM.from_pretrained(iter_path, torch_dtype=torch.bfloat16, use_cache=False).cuda()
        saved_args_path = os.path.join(args.model_path, f'iter_{args.resume_iter:06d}', 'args.jsonl')
        saved_args = json.load(open(saved_args_path))
        saved_iter = saved_args['iter']
        assert saved_iter == args.resume_iter, f'Saved iteration {saved_iter} does not match resume iteration {args.resume_iter}'

        resume_step = saved_iter
    else:
        pprint(f'Loading model from scratch')
        model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, use_cache=False).cuda()

    if not args.use_hybrid:
        model = FSDP(model, auto_wrap_policy=granular_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=torch.cuda.current_device())
    else:
        assert LOCAL_GROUP is not None and CROSS_GROUP is not None, f'LOCAL_GROUP and CROSS_GROUP must be set when using hybrid FSDP'
        model = FSDP(model, auto_wrap_policy=granular_wrap_policy, sharding_strategy=ShardingStrategy.HYBRID_SHARD, device_id=torch.cuda.current_device(), process_group=(LOCAL_GROUP, CROSS_GROUP))

    if args.use_checkpointing:
        pprint(f'Using activation checkpointing')
        def checkpoint_llama_blocks(submodule: torch.nn.Module) -> bool:
            return isinstance(submodule, LlamaDecoderLayer)
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,                      
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=checkpoint_llama_blocks,
        )

    pprint("Main model:", model)

    real_bs = args.micro_batch_size * WORLD_SIZE
    assert args.global_batch_size % real_bs == 0
    num_steps = args.hessian_train_seq // real_bs

    def get_model_hessians(hook_names: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
        all_hooks = register_model_hooks(model, scale=1.0, hook_names=hook_names)
        hessian_iter = iter(train_dataloader)
        with torch.no_grad():
            model.train()
            for j in tqdm(range(num_steps)):
                batch = next(hessian_iter)
                model(**batch)
        torch.cpu.synchronize()
        torch.cuda.synchronize()
        start_time = time.time()
        hook_results = extract_hook_hessians(all_hooks)
        end_time = time.time()
        del all_hooks
        torch.cuda.empty_cache()
        pprint(f"Time taken by extract_hook_hessians: {end_time - start_time} seconds")
        return hook_results

    pprint(f'Loading reference model')
    model_path = args.ref_model
    model_ref = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_cache=False).cuda()
    if not args.use_hybrid:
        model_ref = FSDP(model_ref,  auto_wrap_policy=granular_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=torch.cuda.current_device())
    else:
        assert LOCAL_GROUP is not None and CROSS_GROUP is not None, f'LOCAL_GROUP and CROSS_GROUP must be set when using hybrid FSDP'
        model_ref = FSDP(model_ref,  auto_wrap_policy=granular_wrap_policy, sharding_strategy=ShardingStrategy.HYBRID_SHARD, device_id=torch.cuda.current_device(), process_group=(LOCAL_GROUP, CROSS_GROUP))


    for param in model_ref.parameters():
        param.requires_grad = False
    pprint("Reference model:", model_ref)

    pprint(f'Creating optimizer')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # LR Scheduler will be initialized after quant_groups is defined
    scheduler = None

    if args.resume_iter is not None:
        all_optim_paths = glob.glob(os.path.join(args.model_path, f'iter_{args.resume_iter:06d}', 'optimizer_*.pt'))
        if len(all_optim_paths) == 0:
            pprint(f'WARNING: No optimizer found at {args.model_path}/iter_{args.resume_iter:06d}/optimizer_*.pt, will continue without load')
        else:
            assert len(all_optim_paths) == WORLD_SIZE, f'Expected {WORLD_SIZE} optimizers at {args.model_path}/iter_{args.resume_iter:06d}/optimizer_*.pt, found {len(all_optim_paths)}'
            rank_optim_name = f'optimizer_{RANK:02d}.pt'
            optim_path = os.path.join(args.model_path, f'iter_{args.resume_iter:06d}', rank_optim_name)
            pprint(f'Loading optimizer state at iter {args.resume_iter} from {optim_path}', allrank=True)
            optimizer.load_state_dict(torch.load(optim_path))

    train_iter = iter(train_dataloader)
    quant_groups = grouped_modules_to_quantize(model, args.quant_strategy)
    assert len(quant_groups) > 0, f'No quant groups found for {args.quant_strategy}'
    pprint(f'Grouped modules to quantize: {quant_groups}')

    def lr_warmup_fn(step: int) -> float:
        if step < 10:
            return step / 10.0 
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup_fn)

    if RANK == 0:
        writer : Any = SummaryWriter(log_dir=os.path.join(args.out_path, 'tensorboard_logs')) # type: ignore[no-untyped-call]
        
        hparams = {
            'model_path': args.model_path,
            'ref_model': args.ref_model,
            'train_data': args.train_data,
            'seq_len': args.seq_len,
            'global_batch_size': args.global_batch_size,
            'hessian_train_seq': args.hessian_train_seq,
            'valid_seq': args.valid_seq,
            'micro_batch_size': args.micro_batch_size,
            'total_train_steps': args.total_train_steps,
            'lr': args.lr,
            'hessian_corr': args.hessian_corr,
            'quant_strategy': args.quant_strategy,
            'disable_ldlq': args.disable_ldlq,
            'checkpoint_iters': args.checkpoint_iters,
            'use_checkpointing': args.use_checkpointing,
            'use_hybrid': args.use_hybrid,
            'resume_iter': args.resume_iter,
        }
        writer.add_hparams(hparams, {})

    train_step = 0
    pprint(f'Starting train')
    if resume_step > 0:
        # Dummy forward to get FSDP stuff set, summon_full_params for freezing might not work in the resume case where no forward happens before freezing
        batch = next(iter(train_dataloader))
        with torch.no_grad():
            model(**batch)
        torch.cuda.synchronize()    
    
    if len(quant_groups) == 1:
        pprint('WARNING: Only one quantization group, no self-distillation will be done')
        steps_per_quantization_chunk = 0
    else:
        steps_per_quantization_chunk = args.total_train_steps // (len(quant_groups) - 1)
        if args.total_train_steps % (len(quant_groups) - 1) != 0:
            pprint(f"WARNING: total_train_steps ({args.total_train_steps}) is not perfectly divisible by the number of quantization groups ({len(quant_groups)}). Steps will be rounded down.")
        pprint(f'Skipping self-distillation before first quantization and doing {steps_per_quantization_chunk} steps of self-distillation for each remaining quantization group')

    for quant_group_idx, quant_group in enumerate(quant_groups):
        model.train()

        for step in range(steps_per_quantization_chunk):
            if quant_group_idx == 0:
                break
            train_step_start_time = time.time()
            if train_step < resume_step:
                num_microbatches = args.global_batch_size // real_bs
                for i in tqdm(range(num_microbatches)):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_dataloader)
                        batch = next(train_iter)
                        n_repeats += 1
                        pprint(f'Repeating dataloader {n_repeats} times')
                train_step += 1
                continue
            
            model.zero_grad()
            num_microbatches = args.global_batch_size // real_bs
            total_loss = 0.0
            total_kl = 0.0
            total_ref_entropy = 0.0

            for i in tqdm(range(num_microbatches)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)
                    n_repeats += 1
                    pprint(f'Repeating dataloader {n_repeats} times')

                with torch.no_grad():
                    ref_outputs = model_ref(**batch, output_hidden_states=False)

                outputs = model(**batch, output_hidden_states=False)

                ref_logits = ref_outputs.logits
                logits = outputs.logits
                ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1, dtype=torch.float32).to(logits.dtype)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), ref_probs.view(-1, logits.size(-1)), reduction='mean')
                loss.backward() # type: ignore[no-untyped-call]
                total_loss += loss.item()
                
                with torch.no_grad():
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    log_ref_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    kl = torch.nn.functional.kl_div(log_probs.view(-1, logits.size(-1)), log_ref_probs.view(-1, logits.size(-1)), reduction='batchmean', log_target=True)

                    ref_entropy = torch.nn.functional.cross_entropy(log_ref_probs.view(-1, ref_probs.size(-1)), ref_probs.view(-1, ref_probs.size(-1)), reduction='mean')
                    total_kl += kl.item()
                    total_ref_entropy += ref_entropy.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_step_end_time = time.time()
            train_step_time = train_step_end_time - train_step_start_time
            train_step += 1

            total_loss /= num_microbatches
            total_kl /= num_microbatches
            total_ref_entropy /= num_microbatches

            eval_start_time = time.time()
            model.eval()
            model_ref.eval()

            num_eval_microbatches = args.valid_seq // real_bs
            with torch.no_grad():
                eval_iter = iter(eval_dataloader)
                valid_loss = 0.0
                valid_kl = 0.0
                valid_ref_entropy = 0.0
                valid_ppl = 0.0
                valid_ref_ppl = 0.0
                
                for i in range(num_eval_microbatches):
                    batch = next(eval_iter)

                    ref_outputs = model_ref(**batch, output_hidden_states=False)
                    ref_logits = ref_outputs.logits

                    outputs = model(**batch, output_hidden_states=False)
                    logits = outputs.logits

                    ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1, dtype=torch.float32).to(logits.dtype)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), ref_probs.view(-1, logits.size(-1)), reduction='mean')
                    valid_loss += loss.item()

                    # Get KL and ref entropy
                    log_model_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    log_ref_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    kl = torch.nn.functional.kl_div(log_model_probs.view(-1, logits.size(-1)), log_ref_probs.view(-1, logits.size(-1)), reduction='batchmean', log_target=True)

                    ref_entropy = torch.nn.functional.cross_entropy(log_ref_probs.view(-1, ref_probs.size(-1)), ref_probs.view(-1, ref_probs.size(-1)), reduction='mean')

                    valid_kl += kl.item()
                    valid_ref_entropy += ref_entropy.item()

                    # Get input_ids and shift them by 1 to get target_ids
                    target_ids = batch['input_ids'].cuda()[:, 1:]
                    # Get logits and truncate to target_ids length
                    ppl_logits = logits.view(-1, logits.size(-1))[:target_ids.numel()]
                    loss_ppl = torch.nn.functional.cross_entropy(ppl_logits, target_ids.view(-1), reduction='mean')
                    ppl = torch.exp(loss_ppl)
                    valid_ppl += ppl.item()

                    ppl_ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:target_ids.numel()]
                    loss_ref_ppl = torch.nn.functional.cross_entropy(ppl_ref_logits, target_ids.view(-1), reduction='mean')
                    ref_ppl = torch.exp(loss_ref_ppl)
                    valid_ref_ppl += ref_ppl.item()

                    valid_ppl += ppl.item()
                    valid_ref_ppl += ref_ppl.item() 

            valid_loss /= num_eval_microbatches
            valid_kl /= num_eval_microbatches
            valid_ref_entropy /= num_eval_microbatches
            valid_ppl /= num_eval_microbatches
            valid_ref_ppl /= num_eval_microbatches
            
            # We care about reproducibility across different parallelism sizes for valid, so reduce the metrics
            def reduce_metric(metric_value: float) -> float:
                metric_tensor = torch.tensor(metric_value, device='cuda')
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
                return metric_tensor.item()
            
            valid_loss = reduce_metric(valid_loss)
            valid_kl = reduce_metric(valid_kl)
            valid_ref_entropy = reduce_metric(valid_ref_entropy)
            valid_ppl = reduce_metric(valid_ppl)
            valid_ref_ppl = reduce_metric(valid_ref_ppl)

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            pprint(f'Step {train_step} global loss {total_loss} global kl {total_kl} global ref entropy {total_ref_entropy} train_time {train_step_time} eval_time {eval_time}| valid global loss {valid_loss} valid global kl {valid_kl} valid global ref entropy {valid_ref_entropy} valid ppl {valid_ppl} valid ref ppl {valid_ref_ppl}')

            if RANK == 0:
                writer.add_scalar('Loss/Train/Global', total_loss, train_step)
                writer.add_scalar('Loss/Train/Global_kl', total_kl, train_step)
                writer.add_scalar('Loss/Train/Global_ref_entropy', total_ref_entropy, train_step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], train_step)
                writer.add_scalar('Loss/Valid/Global', valid_loss, train_step)
                writer.add_scalar('Loss/Valid/Global_kl', valid_kl, train_step)
                writer.add_scalar('Loss/Valid/Global_ref_entropy', valid_ref_entropy, train_step)
                writer.add_scalar('Loss/Valid/PPL', valid_ppl, train_step)
                writer.add_scalar('Loss/Valid/Ref_PPL', valid_ref_ppl, train_step)
                writer.add_scalar('Time/Train/Step', train_step_time, train_step)
                writer.add_scalar('Time/Eval/Step', eval_time, train_step)

            if train_step % args.checkpoint_iters == 0 and train_step > 0:
                save_model(model, args.out_path, train_step, vars(args) | {'n_repeats': n_repeats, 'iter': train_step})

        model.zero_grad()

        quant_group_names = [name for name, _ in quant_group]
        quant_group_types = [q_type for _, q_type in quant_group]
        
        if train_step < resume_step:
            pprint(f'Skipping quantization since resume step is not yet reached ({train_step} < {resume_step}), freezing directly')        
            for name, module in model.named_modules():
                if name in quant_group_names:
                    for param_name, param in module.named_parameters():
                        print(f'Freezing {name} | {param_name}')
                        param.requires_grad = False
                        param.grad = None
            continue

        quant_start_time = time.time()
        pprint(f'Finished training phase, getting hessians for quantization')

        hook_names = [name.replace('k_proj', 'q_proj').replace('v_proj', 'q_proj').replace('gate_proj', 'up_proj') for name in quant_group_names]
        hook_results = get_model_hessians(hook_names)
        pprint(f'Got hessians, quantizing')

        for name, module in model.named_modules():
            if name not in quant_group_names:
                continue
            
            q_type = quant_group_types[quant_group_names.index(name)]
            if q_type is None:
                raise ValueError(f'No quantization type found for {name}')
            
            hook_name = name.replace('k_proj', 'q_proj').replace('v_proj', 'q_proj').replace('gate_proj', 'up_proj')
            if hook_name in hook_results and not args.disable_ldlq:
                assert not 'embed_tokens' in name and not 'lm_head' in name, f'Cannot quantize embed_tokens or lm_head with LDLQ, so they should not be hooked'
                H, mu, ct = hook_results[hook_name]
                H = regularize_H(H, H.shape[0], sigma_reg=args.hessian_corr)
                with FSDP.summon_full_params(module), torch.no_grad():
                    W_model = module._fsdp_wrapped_module.weight.data
                    W = torch.zeros_like(W_model, device='cuda')
                    W.copy_(W_model)

                    quant_primitive = get_quant_primitive(q_type)
                    
                    pprint(f'Quantizing module {name} with shape {W.shape}')
                    hatW, quant_state = quantize_LDLQ(W, H, device='cuda', quant_primitive=quant_primitive)
                    dbg(hatW, f'quant-{name}-hatW')
                    module._fsdp_wrapped_module.weight.copy_(hatW.to())
                    pprint(f'Quantized module {name} with shape {hatW.shape}. Freezing it now. ')
                    pprint(f'Saving quant state to {args.out_path}/quant_states/{name}.pt')
                    os.makedirs(os.path.join(args.out_path, 'quant_states'), exist_ok=True)
                    torch.save(quant_state, os.path.join(args.out_path, 'quant_states', f'{name}.pt'))

            else:
                if not ('lm_head' in name or 'embed_tokens' in name):
                    assert args.disable_ldlq, f'Module {name} is not a linear layer or lm_head or embed_tokens, but it wasnt hooked and LDLQ is enabled'
                with FSDP.summon_full_params(module), torch.no_grad():
                    W_model = module._fsdp_wrapped_module.weight.data
                    W = torch.zeros_like(W_model, device='cuda')
                    W.copy_(W_model)

                    quant_primitive = get_quant_primitive(q_type)
                    
                    pprint(f'Quantizing module {name} with shape {W.shape}, without hook')

                    hatW, quant_state = quant_primitive(W, True)
                    dbg(hatW, f'dequant-{name}-hatW')
                    module._fsdp_wrapped_module.weight.copy_(hatW)
                    pprint(f'Quantized module {name} with shape {hatW.shape}. Freezing it now. ')
                    pprint(f'Saving quant state to {args.out_path}/quant_states/{name}.pt')
                    os.makedirs(os.path.join(args.out_path, 'quant_states'), exist_ok=True)
                    torch.save(quant_state, os.path.join(args.out_path, 'quant_states', f'{name}.pt'))

            for param_name, param in module.named_parameters():
                print(f'Freezing {name} | {param_name}')
                param.requires_grad = False
                param.grad = None
                
        quant_end_time = time.time()
        quant_time = quant_end_time - quant_start_time
        pprint(f'Finished quantization, total time {quant_time}')
        pprint(f'Dropping frozen params from optimizer')
        
        for g in optimizer.param_groups:
            pprint(f'Number of params in group: before dropping frozen params {len(g["params"])}')
            g["params"][:] = [p for p in g["params"] if p.requires_grad]
            pprint(f'Number of params in group: after dropping frozen params {len(g["params"])}')
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]

    save_model(model, args.out_path, train_step, vars(args) | {'n_repeats': n_repeats, 'iter': train_step})

    cleanup_dist()


if __name__ == '__main__':
    pprint(f'Running on {socket.gethostname()}')
    args = parser.parse_args()
    init_dist()
    os.makedirs(args.out_path, exist_ok=True)

    assert len(os.listdir(args.out_path)) == 0, f"Output directory {args.out_path} must be empty"

    dist.barrier()

    torch.manual_seed(0)
    main(args)
