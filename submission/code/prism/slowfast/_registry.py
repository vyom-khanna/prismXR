class _Reg:
    def register(self):
        return lambda c: c


MODEL_REGISTRY = _Reg()
