import torch
from quantizer.smooth_quantizer_fake import SmoothQuantizer
class OPTQuantizer(SmoothQuantizer):
    def __init__(self,rep_file, w_bit=16, a_bit=16, w_group_size=64, a_group_size=128, w_granularity="per_group", a_granularity="per_token", w_zero_point=True, a_zero_point=False):
        super().__init__(rep_file,w_bit, a_bit, w_group_size, a_group_size, w_granularity, a_granularity, w_zero_point, a_zero_point)
        self.quant_ignore=["lm_head"]

    def wa_quantize_model(self, model):
        from module.qlinear.qlinear import WALinear
        from utils.utils import get_module_by_name_suffix

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                if name == "model.decoder.layers.31.self_attn.q_proj":
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
        from module.qlinear.qlinear_fake import WALinear
        from utils.utils import get_module_by_name_suffix

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and all([ignore_name not in name for ignore_name in self.quant_ignore]):
                # print("origin: ", getattr(module, 'weight').shape)
                # print('module name: ', name)
                if name == "model.decoder.layers.31.self_attn.q_proj":
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
                # print("after: ", getattr(father_module, name.split(".")[-1]))
                torch.cuda.empty_cache()
        return model
