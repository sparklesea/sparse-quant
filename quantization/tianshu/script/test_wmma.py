import torch

from quant import test_wmma_float, test_wmma_32, test_wmma

import argparse

### test settings
Dim = [[16, 16]]

Seqlens = [16]

Batchsizes = [1]
Configs = [[bs, len, indim, outdim] for len in Seqlens for indim, outdim in Dim for bs in Batchsizes]

def apply_index(weight):
    idx = [
        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
        [ 4.,  5.,  6.,  7.,  0.,  1.,  2.,  3., 12., 13., 14., 15.,  8.,  9., 10., 11.],
        [ 4.,  5.,  6.,  7.,  0.,  1.,  2.,  3., 12., 13., 14., 15.,  8.,  9., 10., 11.],
        [ 4.,  5.,  6.,  7.,  0.,  1.,  2.,  3., 12., 13., 14., 15.,  8.,  9., 10., 11.],
        [ 4.,  5.,  6.,  7.,  0.,  1.,  2.,  3., 12., 13., 14., 15.,  8.,  9., 10., 11.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.],
        [12., 13., 14., 15.,  8.,  9., 10., 11.,  4.,  5.,  6.,  7.,  0.,  1., 2.,  3.],
        [12., 13., 14., 15.,  8.,  9., 10., 11.,  4.,  5.,  6.,  7.,  0.,  1., 2.,  3.],
        [12., 13., 14., 15.,  8.,  9., 10., 11.,  4.,  5.,  6.,  7.,  0.,  1., 2.,  3.],
        [12., 13., 14., 15.,  8.,  9., 10., 11.,  4.,  5.,  6.,  7.,  0.,  1., 2.,  3.]
    ]

    idx = torch.tensor(idx).to(torch.long).cuda()
    weight = torch.gather(weight, 1, idx)
    return weight.transpose(0, 1).contiguous()

def ref_func():
    torch_quant_output = torch.matmul(A, B)
    return torch_quant_output

def inf_func():
    test_wmma_float(A, B, C)
    infini_output = C
    return infini_output

def inf_func_2():
    test_wmma(A, B, C)
    infini_output = C
    return infini_output

def inf_func_32():
    test_wmma_32(A, B, C)
    infini_output = C
    return infini_output.to(torch.float16)

for BS,SEQ,DIM1,DIM2 in Configs:
    # B_temp = [[i for i in range(16)] for _ in range (16)]
    # print(B_temp)
    # A = torch.eye(16, dtype=torch.float32, device="cuda").repeat(1, 4)
    A = torch.randn((16, 64), dtype=torch.float32, device="cuda")
    # A = torch.randn((16, 16), dtype=torch.float32, device="cuda")
    # B = torch.randn((16, 16), dtype=torch.float32, device="cuda")
    B = torch.randn((64, 16), dtype=torch.float32, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
    ref_out = ref_func()
    print(A)
    A = A.transpose(0,1).reshape(4, 16, 16).permute(0, 2, 1).contiguous()
    print(A)
    # B = apply_index(B)
    infini_out = inf_func()
    # infini_out = apply_index(infini_out)
    # infini_out = inf_func_2()

    # A = torch.ones((16, 32), dtype=torch.float16, device="cuda")
    # # B = torch.ones((32, 16), dtype=torch.float16, device="cuda")
    # B = torch.tensor(B_temp, dtype=torch.float16, device="cuda")
    # C = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
    # ref_out = ref_func()
    # B = B.transpose(0,1).reshape(4,-1).transpose(0,1).reshape(-1,128).transpose(0,1).contiguous()
    # infini_out = inf_func_32()

    # print("weight: ", weight_fp16)
    print("ref_out: ", ref_out)
    print("infini_opt: ", infini_out)

    all_close = torch.allclose(ref_out, infini_out, atol=2e-2, rtol=2e-2)
    max_bias = (abs(ref_out - infini_out)).max()

    print(str(BS).ljust(len('batch')) + "  " +
        str(SEQ).ljust(len('seqlen')) + "  " +
        str(DIM1).ljust(len('input_dim')) + "  " +
        str(DIM2).ljust(len('output_dim')) + "  " +
        str(bool(all_close)).ljust(len('all_close')) + "  " +
        "{:.4f}".format(max_bias).ljust(len('max_bias')))