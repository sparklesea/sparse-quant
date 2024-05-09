class BaseQuantizer:
    def __init__(self):
        pass

    def quantize_model(self, model):
        pass

    def __call__(self, model):
        return self.quantize_model(model)
