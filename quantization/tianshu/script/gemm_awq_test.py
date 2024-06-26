import torch

from quant_dequant_tianshu import quant_tianshu as quant_func

# from quant import gemm_awq_ut, gemm_awq_residual_ut, gemm_awq_silu_dot_ut
from quant import gemm_awq_ut

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--benchmark', default=False, action='store_true')
parser.add_argument('--MP', type=int, default=1)
parser.add_argument('--algo_id', type=int, default=0)
args = parser.parse_args()

MP = args.MP

### benchmark settings
WARM_UP = args.warmup
REP = args.rep
BENCHMARK_FLAG = args.benchmark

### test settings
GROUP_SIZE = 128
# Dim = [[4096 // MP, 4096], [5120 // MP, 5120], [11008 // MP, 4096], [13696 // MP, 5120]] if args.algo_id < 2 else \
#     [[4096, 11008 // MP], [5120, 13696 // MP]]
Dim = [[4096 // MP, 4096]]

Seqlens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

Batchsizes = [1]
Configs = [[bs, len, indim, outdim] for len in Seqlens for indim, outdim in Dim for bs in Batchsizes]

algo_id = args.algo_id

if (algo_id < 0 or algo_id > 3):
    raise Exception("algo_id, must be in range of [0, 3]")

def apply_index(weight):
    org_shape = weight.shape
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
    idx = idx.reshape(1, -1).repeat(weight.numel()/64).reshape(-1, 16)
    weight = torch.gather(weight.reshape(-1, 16), 1, idx)
    return weight.reshape(org_shape).transpose(0, 1).contiguous()

def ref_func():
    torch_quant_output = torch.matmul(x, weight_fp16.t())
    # if (algo_id == 1):
    #     torch_quant_output += r
    # if (algo_id == 2 or algo_id == 3):
    #     torch_quant_output2 = torch.matmul(x, weight_fp162.t())
    #     torch_quant_output = torch.nn.functional.silu(torch_quant_output) * torch_quant_output2
    return torch_quant_output

def inf_func():
    # if (algo_id == 0):
    infini_output = gemm_awq_ut(x, qweight, zeros_scales,
                            BS * SEQ, DIM2, DIM1, GROUP_SIZE)
    # infini_out = apply_index(infini_out)
    
    return infini_output

AlgoList = [
    "gemm awq",
    "gemm awq residual",
    "gemm awq silu dot",
    "gemm awq packed silu dot",
]

print("select: ", AlgoList[algo_id])

if not BENCHMARK_FLAG:
    print("batch  seqlen  input_dim  output_dim  all_close  max_bias")
else:
    print("batch  seqlen  input_dim  output_dim  ref_dur(ms)  infini_dur(ms)  speedup")

for BS,SEQ,DIM1,DIM2 in Configs:

    x = torch.ones((BS, SEQ, DIM1), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    # x = torch.ones((BS, SEQ, DIM1), dtype=torch.float16, device="cuda")
    linear_layer = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)
    # torch.nn.init.constant_(linear_layer.weight, 1)

    # weight = torch.randn(DIM1, DIM2, device="cuda", dtype=torch.float16)
    # linear_layer2 = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)
    # r = torch.empty((BS, SEQ, DIM2), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    qweight, zeros_scales, weight_fp16 = quant_func(linear_layer.weight, DIM1, DIM2, GROUP_SIZE)
    # qweight2, zeros_scales2, weight_fp162 = quant_func(linear_layer2.weight, DIM1, DIM2, GROUP_SIZE)
    # if (algo_id == 3):
    #     w13 = torch.cat((linear_layer.weight, linear_layer2.weight), dim=0).contiguous()
    #     qweight13, zeros_scales13, _ = quant_func(w13, DIM1, DIM2 * 2, GROUP_SIZE)
    
    ref_out = ref_func()
    infini_out = inf_func()

    # print("weight: ", weight_fp16)
    print("ref_out: ", ref_out)
    print("infini_opt: ", infini_out)

    all_close = torch.allclose(ref_out, infini_out.reshape((BS, SEQ, DIM2)), atol=2e-2, rtol=2e-2)
    max_bias = (abs(ref_out - infini_out.reshape((BS, SEQ, DIM2)))).max()

    if not BENCHMARK_FLAG:
        print(str(BS).ljust(len('batch')) + "  " +
            str(SEQ).ljust(len('seqlen')) + "  " +
            str(DIM1).ljust(len('input_dim')) + "  " +
            str(DIM2).ljust(len('output_dim')) + "  " +
            str(bool(all_close)).ljust(len('all_close')) + "  " +
            "{:.4f}".format(max_bias).ljust(len('max_bias')))
    else:
        ### benchmarking
        for _ in range(WARM_UP):
            ref_out = ref_func()

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            ref_out = ref_func()
            end_event[i].record()
        torch.cuda.synchronize()
        ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


        for _ in range(WARM_UP):
            infini_out = inf_func()

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            infini_out = inf_func()
            end_event[i].record()
        torch.cuda.synchronize()
        infini_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

        print(str(BS).ljust(len('batch')) + "  " +
            str(SEQ).ljust(len('seqlen')) + "  " +
            str(DIM1).ljust(len('input_dim')) + "  " +
            str(DIM2).ljust(len('output_dim')) + "  " +
            "{:.4f}".format(torch.mean(ref_dur).item()).ljust(len('ref_dur(ms)')) + "  " +
            "{:.4f}".format(torch.mean(infini_dur).item()).ljust(len('infini_dur(ms)')) +  "  " +
            "{:.4f}".format(torch.mean(ref_dur).item() / torch.mean(infini_dur).item()).ljust(len('speedup'))) 
