import torch
from torch import nn
from functools import partial
from utils.quant_funcs import pseudo_quantize_tensor


class WALinear(nn.Module):
    def __init__(self, in_features, out_features, w_config, a_config, bias=True, quantize_output=False, dev="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_config = w_config
        self.a_config = a_config

        self.register_buffer("weight", torch.zeros(self.out_features, self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        if bias:
            self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_buffer("bias", None)

        self.act_quant = partial(pseudo_quantize_tensor, **self.a_config)

        if quantize_output:
            self.output_quant = self.act_quant
        else:
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WALinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, w_config, a_config, quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = WALinear(module.in_features, module.out_features, w_config, a_config, module.bias is not None, quantize_output=quantize_output, dev=module.weight.device)
        new_module.weight = pseudo_quantize_tensor(module.weight, inplace=True, **w_config)

        if module.bias is not None:
            new_module.bias = module.bias
        del module
        return new_module

    def __repr__(self):
        return "W{}A{}Linear".format(self.w_config["n_bit"], self.a_config["n_bit"])
    
