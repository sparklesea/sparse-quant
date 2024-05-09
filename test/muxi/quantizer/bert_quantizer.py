import torch
from quantizer.rtn_quantizer import RTNQuantizer
class BERTQuantizer(RTNQuantizer):
    def __init__(self, w_bit=16, a_bit=16, kv_bit=16, w_group_size=128, a_group_size=128, kv_group_size=128, w_granularity="per_group", a_granularity="per_token", kv_granularity="per_group", w_zero_point=True, a_zero_point=False, kv_zero_point=True):
        super().__init__(w_bit, a_bit, kv_bit, w_group_size, a_group_size, kv_group_size, w_granularity, a_granularity, kv_granularity, w_zero_point, a_zero_point, kv_zero_point)
        self.quant_ignore=['cls']

    def wa_quantize_model(self, model):
        from module.qlinear.qlinear import WALinear
        from utils.utils import get_module_by_name_suffix

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                if "pooler" in name and "dense" in name:
                    w_bit=2
                else:
                    w_bit=self.w_config["n_bit"]
                new_w_config=self.gen_q_config(**self.w_config)
                new_w_config["n_bit"]=w_bit
                new_linear = WALinear.from_float(module, new_w_config, self.a_config, quantize_output=False)
                father_module = get_module_by_name_suffix(model, ".".join(name.split(".")[:-1]))
                setattr(father_module, name.split(".")[-1], new_linear)
                del new_linear, module
                torch.cuda.empty_cache()
        return model

    def w_quantize_model(self, model):
        from module.qlinear.qlinear import WALinear
        from utils.utils import get_module_by_name_suffix

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                if "pooler" in name and "dense" in name:
                    w_bit=2
                else:
                    w_bit=self.w_config["n_bit"]
                # w_bit=self.w_config["n_bit"]
                new_w_config=self.gen_q_config(**self.w_config)
                new_w_config["n_bit"]=w_bit
                new_linear = WALinear.from_float(module, new_w_config, self.a_config, quantize_output=False)
                father_module = get_module_by_name_suffix(model, ".".join(name.split(".")[:-1]))
                setattr(father_module, name.split(".")[-1], new_linear)
                del new_linear, module
                torch.cuda.empty_cache()
        return model
