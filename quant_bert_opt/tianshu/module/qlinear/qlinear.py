import torch
from torch import nn
from functools import partial
from utils.quant_funcs import pseudo_quantize_tensor, quant_tianshu
from quant import gemm_awq_ut


class WALinear(nn.Module):
    def __init__(self, in_features, out_features, w_config = None, a_config = None,
                 w_bit = 4, a_bit = 16, group_size = 128,
                 bias=True, quantize_output=False, dev="cuda", dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_config = w_config
        self.a_config = a_config
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size

        # print("origin weight shape: ", (self.out_features, self.in_features * w_bit // 8))
        # print("origin zero_scales shape: ", (self.out_features, self.in_features * 2 // group_size))

        self.register_buffer("zeros_scales", torch.zeros(self.out_features, self.in_features * 2 // group_size, dtype=torch.float16, requires_grad=False, device=dev))

        self.register_buffer("weight", torch.zeros(self.out_features, self.in_features * w_bit // 8, dtype=dtype, requires_grad=False, device=dev))
        # self.register_buffer("weight", torch.zeros(self.out_features, self.in_features * self.w_bit // 8, dtype=dtype, requires_grad=False, device=dev))
        if bias:
            # self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=dtype, requires_grad=False, device=dev))
            self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_buffer("bias", None)

        # self.act_quant = partial(pseudo_quantize_tensor, **self.a_config)

        # if quantize_output:
        #     self.output_quant = self.act_quant
        # else:
        #     self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WALinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    # @torch.no_grad()
    # def forward(self, x):
    #     q_x = self.act_quant(x)
    #     print(self.weight)
    #     y = torch.functional.F.linear(x, self.weight, self.bias)
    #     q_y = self.output_quant(y)
    #     return q_y

    @torch.no_grad()
    def forward(self, x):
        if self.w_bit == 2:
            temp_weight = self.convert_2bit_to_4bit(self.weight)
        else:
            temp_weight = self.weight

        # if torch.any(torch.isnan(x)):
        #     exit(0)
        # print("========================================================")
        # print("x: ", x, x.shape, x.dtype)
        # print("temp_weight: ", temp_weight, temp_weight.shape, temp_weight.dtype)
        # print("zeros_scales: ", self.zeros_scales, self.zeros_scales.shape, self.zeros_scales.dtype)

        y = gemm_awq_ut(x,temp_weight,self.zeros_scales,x.shape[-2],
                        self.out_features, self.in_features, self.group_size) + self.bias
        # print("y: ", y, y.shape, y.dtype)
        # torch.save(x, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/x.pt')
        # torch.save(temp_weight, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/weight.pt')
        # torch.save(self.zeros_scales, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/zeros_scales.pt')
        # if torch.any(torch.isnan(y)):
            # torch.save(x, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/x.pt')
            # torch.save(temp_weight, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/weight.pt')
            # torch.save(self.zeros_scales, '/home/huangshan/huangshan/project/sparse-quant/quantization/nvidia/script/zeros_scales.pt')
            # exit(0)
        # print("========================================================")
        return y

    @staticmethod
    def from_float(module, w_config, a_config, quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        # print("origin shape: ", module.weight.shape)
        new_module = WALinear(module.in_features, module.out_features, w_config, a_config, w_bit=w_config['n_bit'],
                              bias=module.bias is not None, quantize_output=quantize_output, dev=module.weight.device, dtype=torch.uint8)
        # new_module.weight, new_module.zeros_scales = pseudo_quantize_tensor(module.weight, inplace=True, **w_config)
        new_module.weight, new_module.zeros_scales = quant_tianshu(module.weight, module.in_features, module.out_features,
                                                                n_bit=w_config['n_bit'], group_size=w_config['group_size'])
        print("quant weight shape: ", new_module.weight.shape)
        print("quant zeros_scales shape: ", new_module.zeros_scales.shape)

        if module.bias is not None:
            new_module.bias = module.bias.unsqueeze(0)
        del module
        return new_module

    @staticmethod
    def convert_2bit_to_4bit(qweight_2bit):
        origin_shape = qweight_2bit.shape
        qweight_4bit = torch.empty((origin_shape[0] * origin_shape[1], 2), dtype=torch.uint8, device=qweight_2bit.device)
        qweight_2bit_flatten = qweight_2bit.view(-1)

        qweight_4bit[:, 0] = (qweight_2bit_flatten[:] & 0xC0) >> 2
        qweight_4bit[:, 0] += ((qweight_2bit_flatten[:] & 0x30) >> 4)
        qweight_4bit[:, 1] = (qweight_2bit_flatten[:] & 0x0C) << 2
        qweight_4bit[:, 1] += (qweight_2bit_flatten[:] & 0x03)
        return qweight_4bit.reshape(origin_shape[0], origin_shape[1] * 2)

    def __repr__(self):
        return "W{}A{}Linear({}, {})".format(self.w_bit, self.a_bit, self.in_features, self.out_features)
    
