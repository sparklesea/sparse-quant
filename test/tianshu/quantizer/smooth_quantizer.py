import types
import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm
from transformers.models.rwkv.modeling_rwkv import RwkvFeedForward
from transformers.activations import GELUActivation


from quantizer.rtn_quantizer import RTNQuantizer
from module.scales.scaled_module import ScaledActivation, scaled_rwkv_feedforward
from utils.utils import get_op_by_name, set_op_by_name
from module.qlinear.qlinear import WALinear


class SmoothQuantizer(RTNQuantizer):
    def __init__(self, rep_file=None, w_bit=16, a_bit=16, w_group_size=128, a_group_size=128, w_granularity="per_group", a_granularity="per_token", w_zero_point=True, a_zero_point=False):
        super().__init__(w_bit=w_bit, a_bit=a_bit, w_group_size=w_group_size, a_group_size=a_group_size, w_granularity=w_granularity, a_granularity=a_granularity, w_zero_point=w_zero_point, a_zero_point=a_zero_point)
        if rep_file is not None:
            self.scales = torch.load(rep_file)
            if type(self.scales) == dict:
                self.scales, self.clips = self.scales["scale"], self.scales["clip"]
            else:
                self.clips = []
            self.calibrated = True
        else:
            self.calibrated = False
            self.scales, self.clips = [], []

    def quantize_model(self, model):
        assert self.calibrated == True
        self.apply_scale(model)
        self.apply_clip(model)
        return super().quantize_model(model)

    def apply_scale(self, module, input_feat_dict=None):
        for prev_op_name, layer_names, scales in self.scales:
            prev_op = get_op_by_name(module, prev_op_name)
            layers = [get_op_by_name(module, name) for name in layer_names]

            if isinstance(prev_op, nn.Linear):
                assert len(layers) == 1
                self.scale_fc_fc(prev_op, layers[0], scales)
            elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, MixtralRMSNorm, MistralRMSNorm)):
                self.scale_ln_fcs(prev_op, layers, scales)
            elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
                new_module = ScaledActivation(prev_op, scales)
                set_op_by_name(module, prev_op_name, new_module)
                self.scale_gelu_fc(prev_op, layers[0], scales)
            elif isinstance(prev_op, RwkvFeedForward):
                self.scale_square_fc(prev_op, layers[0], scales)
            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            # apply the scaling to input feat if given; prepare it for clipping
            if input_feat_dict is not None:
                for layer_name in layer_names:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

    @torch.no_grad()
    def apply_clip(self, module):
        for name, max_val in self.clips:
            layer = get_op_by_name(module, name)
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)

    def gen_scales_clips(self, enc, model, dataset_path, n_samples, seqlen):
        pass

    @staticmethod
    @torch.no_grad()
    def scale_ln_fcs(ln, fcs, scales):
        if not isinstance(fcs, list):
            fcs = [fcs]

        scales = scales.to(ln.weight.device)

        ln.weight.div_(scales)
        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @staticmethod
    @torch.no_grad()
    def scale_fc_fc(fc1, fc2, scales):
        assert isinstance(fc1, nn.Linear)
        assert isinstance(fc2, nn.Linear)

        scales = scales.to(fc1.weight.device)

        # fc1.weight.div_(scales.view(-1, 1))
        fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(scales.view(-1))

        fc2.weight.mul_(scales.view(1, -1))

        for p in fc1.parameters():
            assert torch.isnan(p).sum() == 0
        for p in fc2.parameters():
            assert torch.isnan(p).sum() == 0

    @staticmethod
    @torch.no_grad()
    def scale_gelu_fc(gelu, fc, scales):
        assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
        assert isinstance(fc, nn.Linear)

        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    @staticmethod
    @torch.no_grad()
    def scale_square_fc(feed_forward, fc, scales):
        assert isinstance(feed_forward, (RwkvFeedForward))
        assert isinstance(fc, nn.Linear)

        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

        setattr(feed_forward, "scales", scales)
        feed_forward.forward = types.MethodType(scaled_rwkv_feedforward, feed_forward)

        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    def save_scales_clips(self, save_path):
        torch.save({"scale": self.scales, "clip": self.clips}, save_path)
