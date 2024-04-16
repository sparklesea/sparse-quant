import torch
from quantizer.base_quantizer import BaseQuantizer
from huangshan.project.quant_bert_opt.utils.quant_funcs import pseudo_quantize_tensor


class RTNQuantizer(BaseQuantizer):
    def __init__(self, w_bit=16, a_bit=16, kv_bit=16, w_group_size=128, a_group_size=128, kv_group_size=128, w_granularity="per_group", a_granularity="per_token", kv_granularity="per_group", w_zero_point=True, a_zero_point=False, kv_zero_point=True):
        self.w_config = self.gen_q_config(w_bit, w_granularity, w_group_size, w_zero_point)
        self.a_config = self.gen_q_config(a_bit, a_granularity, a_group_size, a_zero_point)
        self.kv_config = self.gen_q_config(kv_bit, kv_granularity, kv_group_size, kv_zero_point)
        self.quant_ignore = ["lm_head", "output_layer", "head"]

    def w_quantize_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                try:
                    module.weight.data = pseudo_quantize_tensor(module.weight.data, **self.w_config)
                except:
                    print(f"Failed to quantize {name}")
        return model

    def wa_quantize_model(self, model):
        from module.qlinear.qlinear import WALinear
        from utils.utils import get_module_by_name_suffix

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                new_linear = WALinear.from_float(module, self.w_config, self.a_config, quantize_output=False)
                father_module = get_module_by_name_suffix(model, ".".join(name.split(".")[:-1]))
                setattr(father_module, name.split(".")[-1], new_linear)
                del new_linear, module
                torch.cuda.empty_cache()
        return model

    def quantize_model(self, model):
        if self.w_config["n_bit"] > 0 and self.w_config["n_bit"] < 16 and self.a_config["n_bit"] == 16:
            model = self.w_quantize_model(model)
        if self.w_config["n_bit"] > 0 and self.w_config["n_bit"] < 16 and self.a_config["n_bit"] > 0 and self.a_config["n_bit"] < 16:
            model = self.wa_quantize_model(model)
        return model

    def gen_q_config(self, n_bit, granularity, group_size, zero_point):
        q_config = {}
        q_config["n_bit"] = n_bit
        assert granularity in ["per_tensor", "per_token", "per_channel", "per_group"]
        q_config["granularity"] = granularity
        if granularity == "per_group":
            q_config["group_size"] = group_size
        else:
            q_config["group_size"] = None
        q_config["zero_point"] = zero_point
        return q_config
