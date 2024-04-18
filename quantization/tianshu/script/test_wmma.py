import torch

from quant import test_wmma

import argparse

### test settings
Dim = [[4096, 4096]]

Seqlens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

Batchsizes = [1]
Configs = [[bs, len, indim, outdim] for len in Seqlens for indim, outdim in Dim for bs in Batchsizes]

def ref_func():
    torch_quant_output = torch.matmul(A, B.t()).to(torch.float32)
    return torch_quant_output

def inf_func():
    # if (algo_id == 0):
    test_wmma(A, B, C, BS * SEQ, DIM2, DIM1)
    infini_output = C
    return infini_output

for BS,SEQ,DIM1,DIM2 in Configs:
    A = torch.randn((BS*SEQ, DIM1), dtype=torch.float16, device="cuda")
    B = torch.randn((DIM2, DIM1), dtype=torch.float16, device="cuda")
    C = torch.zeros((BS*SEQ, DIM2), dtype=torch.float32, device="cuda")
    
    ref_out = ref_func()
    infini_out = inf_func()

    # print("weight: ", weight_fp16)
    # print("ref_out: ", ref_out)
    # print("infini_opt: ", infini_out)

    all_close = torch.allclose(ref_out, infini_out, atol=2e-2, rtol=2e-2)
    max_bias = (abs(ref_out - infini_out)).max()

    print(str(BS).ljust(len('batch')) + "  " +
        str(SEQ).ljust(len('seqlen')) + "  " +
        str(DIM1).ljust(len('input_dim')) + "  " +
        str(DIM2).ljust(len('output_dim')) + "  " +
        str(bool(all_close)).ljust(len('all_close')) + "  " +
        "{:.4f}".format(max_bias).ljust(len('max_bias')))