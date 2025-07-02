import ctypes
import os
import multiprocessing as mp
from multiprocessing import Array, Process
from typing import Any, Tuple, Type, Union

import numpy as np
import torch
from cffi import FFI

NUM_THREADS: int = 1

QUANTIZATION_PATH: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ffi = FFI()
header: str = open(f'{QUANTIZATION_PATH}/csrc/fake_cffi_quantize.h').read()
ffi.cdef(header)
lib = ffi.dlopen(f'{QUANTIZATION_PATH}/csrc/quantize.so')

QK_K: int = 256
K_SCALE_SIZE: int = 12

class BLOCK_Q4(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_uint16),
        ("dmin", ctypes.c_uint16),
        ("scales", ctypes.c_uint8 * K_SCALE_SIZE),
        ("qs", ctypes.c_uint8 * (QK_K//2)),
    ]

class BLOCK_Q3(ctypes.Structure):
    _fields_ = [
        ("hmask", ctypes.c_uint8 * 32),
        ("qs", ctypes.c_uint8 * 64),
        ("scales", ctypes.c_uint8 * 12),
        ("d", ctypes.c_uint16),
    ]

class BLOCK_Q6(ctypes.Structure):
    _fields_ = [
        ("ql", ctypes.c_uint8 * (QK_K//2)),
        ("qh", ctypes.c_uint8 * (QK_K//4)),
        ("scales", ctypes.c_int8 * (QK_K//16)),
        ("d", ctypes.c_uint16),
    ]


def quantize_worker(
    start_idx: int, 
    end_idx: int, 
    float_array: np.ndarray, 
    res_array: Any,  
    k: int, 
    shared_array: Any, 
    dtype: str = 'q4_K'
) -> None:
    """Worker function for multiprocess quantization."""
    assert start_idx % QK_K == 0
    assert (end_idx) % QK_K == 0
    if dtype == 'q4_K':
        quantize_func = lib.quantize_row_q4_K_ref
    elif dtype == 'q3_K':
        quantize_func = lib.quantize_row_q3_K_ref
    elif dtype == 'q6_K':
        quantize_func = lib.quantize_row_q6_K_ref
    else:
        raise ValueError(f'Invalid dtype {dtype}')
    c_float_array = ffi.cast("float*", ffi.from_buffer(float_array[start_idx:end_idx]))
    c_res_array = res_array + start_idx // QK_K
    quantize_func(c_float_array, c_res_array, k)


def quantize_tensor(W: torch.Tensor, dtype: str = 'q4_K') -> Tuple[np.ndarray, Tuple[Tuple[int, ...], int]]:
    """Quantize tensor using GGML quantization."""
    global NUM_THREADS
    if NUM_THREADS == 1 or W.numel() < 4096*16: # For small matrices mp actually slows things down
        return quantize_tensor_nomp(W, dtype)
    else:
        return quantize_tensor_mp(W, dtype)

def quantize_tensor_nomp(W: torch.Tensor, dtype: str = 'q4_K') -> Tuple[np.ndarray, Tuple[Tuple[int, ...], int]]:
    """Quantize tensor without multiprocessing."""
    if dtype == 'q4_K':
        block_type: Type[ctypes.Structure] = BLOCK_Q4
        str_type = 'block_q4_K'
    elif dtype == 'q3_K':
        block_type = BLOCK_Q3
        str_type = 'block_q3_K'
    elif dtype == 'q6_K':
        block_type = BLOCK_Q6
        str_type = 'block_q6_K'
    orig_shape = W.shape
    assert len(orig_shape) == 2

    W_np: np.ndarray = W.flatten().detach().cpu().contiguous().numpy()
    if W_np.shape[0] % QK_K != 0:
        raise ValueError("Matrix rows must be multiple of QK_K")

    ne: int = W_np.shape[0]

    nblocks: int = ne // QK_K
    res_array = ffi.new(str_type + "[{}]".format(nblocks))

    np_float_array: np.ndarray = W_np.astype(np.float32)
    float_array = ffi.cast("float*", ffi.from_buffer(np_float_array))

    if dtype == 'q4_K':
        quantize_func = lib.quantize_row_q4_K_ref
    elif dtype == 'q3_K':
        quantize_func = lib.quantize_row_q3_K_ref
    elif dtype == 'q6_K':
        quantize_func = lib.quantize_row_q6_K_ref
    else:
        raise ValueError(f'Invalid dtype {dtype}')

    quantize_func(float_array, res_array, ne)

    res_array_np: np.ndarray = np.frombuffer(ffi.buffer(res_array), dtype=np.uint8).reshape(-1,).copy()

    return res_array_np, (orig_shape, ne)
    
def quantize_tensor_mp(W: torch.Tensor, dtype: str = 'q4_K') -> Tuple[np.ndarray, Tuple[Tuple[int, ...], int]]:
    """Quantize tensor with multiprocessing."""
    if dtype == 'q4_K':
        block_type: Type[ctypes.Structure] = BLOCK_Q4
        str_type = 'block_q4_K'
    elif dtype == 'q3_K':
        block_type = BLOCK_Q3
        str_type = 'block_q3_K'
    elif dtype == 'q6_K':
        block_type = BLOCK_Q6
        str_type = 'block_q6_K'
    else:
        raise ValueError(f'Invalid dtype {dtype}')

    orig_shape = W.shape
    assert len(orig_shape) == 2

    W_np: np.ndarray = W.flatten().detach().cpu().contiguous().numpy()
    if W_np.shape[0] % QK_K != 0:
        raise ValueError("Matrix rows must be multiple of QK_K")

    ne: int = W_np.shape[0]

    nblocks: int = ne // QK_K


    shared_array = Array(ctypes.c_char, nblocks * ctypes.sizeof(block_type))
    shared_array_address: int = ctypes.addressof(shared_array.get_obj())
    res_array = ffi.cast(str_type + " *", shared_array_address)

    np_float_array: np.ndarray = W_np.astype(np.float32)

    block_chunk_size: int = (nblocks + NUM_THREADS - 1) // NUM_THREADS

    processes = []
    for i in range(NUM_THREADS):
        start_idx = i * block_chunk_size * QK_K
        end_idx = min(start_idx//QK_K + block_chunk_size, nblocks) * QK_K
        p = Process(target=quantize_worker, args=(start_idx, end_idx, np_float_array, res_array, end_idx - start_idx, shared_array, dtype))
        processes.append(p)
        p.start()


    for p in processes:
        p.join()
    
    res_shared_addr: int = ctypes.addressof(shared_array.get_obj())
    res_array = ffi.cast(str_type + " *", res_shared_addr)

    res_size: int = ctypes.sizeof(block_type) * nblocks
    res_array_np: np.ndarray = np.frombuffer(ffi.buffer(res_array, size=res_size), dtype=np.uint8).reshape(-1,).copy()

    return res_array_np, (orig_shape, ne)

def dequantize_tensor(
    res_numpy: np.ndarray, 
    metadata: Tuple[Tuple[int, ...], int], 
    dtype: str = 'q4_K'
) -> np.ndarray:
    """Dequantize tensor from GGML quantized format."""
    orig_shape, ne = metadata
    if dtype == 'q4_K':
        str_type = 'block_q4_K'
    elif dtype == 'q3_K':
        str_type = 'block_q3_K'
    elif dtype == 'q6_K':
        str_type = 'block_q6_K'
    else:
        raise ValueError(f'Invalid dtype {dtype}')

    res_array = ffi.cast(str_type + " *", res_numpy.ctypes.data)

    float_array: np.ndarray = np.zeros(ne, dtype=np.float32)
    float_array_ptr = ffi.cast("float*", ffi.from_buffer(float_array))
    
    if dtype == 'q4_K':
        lib.dequantize_row_q4_K(res_array, float_array_ptr, ne)
    elif dtype == 'q3_K':
        lib.dequantize_row_q3_K(res_array, float_array_ptr, ne)
    elif dtype == 'q6_K':
        lib.dequantize_row_q6_K(res_array, float_array_ptr, ne)
    else:
        raise ValueError(f'Invalid dtype {dtype}')
    
    return float_array.reshape(orig_shape)
