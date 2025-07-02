from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np

# Local imports
from .ggml import dequantize_tensor as ggml_dequantize_tensor
from .ggml import quantize_tensor as ggml_quantize_tensor


# LDLQ related functions adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/1d8f873e9a2a8b86b12bb1064c312c5689b77d98/lib/utils/math_utils.py#L14
def block_LDL(H: torch.Tensor, b: int, check_nan: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute block LDL decomposition of matrix H."""
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    L = torch.linalg.cholesky(H)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        raise ValueError('L contains NaNs')

    L = L.reshape(n, n)
    return (L, D.to(DL.device))


def LDLQ_quantize(
    W: torch.Tensor, 
    H: torch.Tensor, 
    L: torch.Tensor, 
    quant_primitive: Any, 
    buf_cols: int = 256, 
    quip_tune_iters: int = 3
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], Dict[str, Any]]]:
    """Perform LDLQ quantization on weight matrix W using proxy Hessian H and Cholesky factor L."""
    codesz = 256
    (m, n) = W.shape
    hatW = torch.zeros(m, n, dtype=H.dtype, device=H.device)
    assert n % buf_cols == 0 and buf_cols % codesz == 0
    buf_size = buf_cols // codesz

    quantized_patches: Dict[Tuple[int, int], Dict[str, Any]] = {}
    prod_cache = torch.zeros(m, n, dtype=W.dtype, device=W.device)
    for cur_col in range(n // codesz, 0, -buf_size):
        b_W = W[:, codesz * (cur_col - buf_size):codesz * cur_col]
        b_hatW = hatW[:,
                        codesz * (cur_col - buf_size):codesz * cur_col]
        b_L = L[codesz * (cur_col - buf_size):codesz * cur_col]
        b_prod = prod_cache[:, codesz * (cur_col - buf_size):codesz *
                            cur_col]
        L_offset = codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_W[:, codesz * i : codesz * (i + 1)] + \
                (b_W[:, codesz * (i + 1):] - b_hatW[:, codesz * (i + 1):]) @ \
                b_L[codesz * (i + 1):, L_offset + codesz * i : L_offset + codesz * (i + 1)] + \
                b_prod[:, codesz * i : codesz * (i + 1)]
            b_hatW[:, codesz * i:codesz *
                    (i + 1)], quant_patch = quant_primitive(
                    WXWX)
            quantized_patches[( (cur_col-buf_size)*codesz + codesz*i, (cur_col-buf_size)*codesz + codesz*(i+1))] = quant_patch
                        
        prod_cache += (b_W - b_hatW) @ b_L

    del b_W, b_hatW, b_L, b_prod, L_offset, prod_cache

    # tune
    for ie in range(quip_tune_iters):
        delta = W - hatW
        for cur_col in range(n // codesz, 0, -buf_size):
            b_hatW = hatW[:, codesz * (cur_col - buf_size):codesz *
                            cur_col]
            b_H = H[:, codesz * (cur_col - buf_size):codesz * cur_col]
            b_delta = delta[:, codesz * (cur_col - buf_size):codesz *
                            cur_col]
            H_offset = codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if codesz > 1:
                    inv = torch.linalg.inv(
                        b_H[H_offset + codesz * i:H_offset + codesz *
                             (i + 1), codesz * i:codesz * (i + 1)])
                else:
                    inv = 1 / b_H[H_offset + i:H_offset + i + 1, i:i + 1]

                WXWX = b_hatW[:, codesz * i : codesz * (i + 1)] + \
                    delta @ b_H[:, codesz * i : codesz * (i + 1)] @ inv

                b_delta[:, codesz * i:codesz *
                        (i + 1)] += b_hatW[:,
                                            codesz * i:codesz * (i + 1)]

                if ie < quip_tune_iters - 1:
                    b_hatW[:,
                            codesz * i:codesz * (i + 1)] = quant_primitive(
                                WXWX,
                                return_qs=False)
                else:
                    b_hatW[:, codesz * i:codesz *
                            (i + 1)], quantized_patch = quant_primitive(
                                WXWX)
                    quantized_patches[( (cur_col-buf_size)*codesz + codesz*i, (cur_col-buf_size)*codesz + codesz*(i+1))] = quantized_patch
                   

                b_delta[:, codesz * i:codesz *
                        (i + 1)] -= b_hatW[:,
                                            codesz * i:codesz * (i + 1)]
        del delta, b_hatW, b_H, b_delta, H_offset

    return hatW, quantized_patches


def quantize_ggml(
    W: torch.Tensor, 
    return_qs: bool = True, 
    dtype: str = 'q4_K'
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Quantize tensor using GGML dtypes."""
    orig_dtype = W.dtype
    orig_device = W.device
    W = W.to(torch.float32)
    quant, metadata = ggml_quantize_tensor(W, dtype=dtype)
    np_weight = ggml_dequantize_tensor(quant, metadata, dtype=dtype)
    weight = torch.from_numpy(np_weight).to(device=orig_device, dtype=orig_dtype)
    if return_qs == False:
        return weight
    quant_metadata = {
        'primitive': dtype,
        'metadata': metadata,
        'quant_state': quant
    }
    return weight, quant_metadata


def quantize_q4_K(W: torch.Tensor, return_qs: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Quantize tensor using Q4_K quantization."""
    return quantize_ggml(W, return_qs=return_qs, dtype='q4_K')


def quantize_q3_K(W: torch.Tensor, return_qs: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Quantize tensor using Q3_K quantization."""
    return quantize_ggml(W, return_qs=return_qs, dtype='q3_K')


def quantize_q6_K(W: torch.Tensor, return_qs: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Quantize tensor using Q6_K quantization."""
    return quantize_ggml(W, return_qs=return_qs, dtype='q6_K')


def quantize_LDLQ(
    W: torch.Tensor, 
    H: torch.Tensor, 
    device: str = 'cpu', 
    quant_primitive: Any = quantize_q3_K
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], Dict[str, Any]]]:
    """Perform LDLQ quantization on weight matrix W using proxy Hessian H."""
    orig_device = W.device
    W = W.to(device)
    H = H.to(device)

    Lh = torch.linalg.cholesky(H)
    L, D = block_LDL(H, 8)
    hatW, quantized_patches = LDLQ_quantize(W, H, L, quant_primitive=quant_primitive)
    hatW = hatW.to(orig_device)
    return hatW, quantized_patches
