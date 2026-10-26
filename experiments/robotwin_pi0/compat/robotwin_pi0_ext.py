"""Project-local compatibility for the pinned RLinf OpenPI/PEFT versions."""
from types import SimpleNamespace

class RuntimeConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()

def build_openpi(cfg, torch_dtype):
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype)
    # PEFT reads to_dict(); RLinf workers continue to use attribute access.
    if isinstance(model.config, SimpleNamespace):
        model.config = RuntimeConfig(**vars(model.config))
    return model

def register():
    from rlinf.models import register_model
    register_model("openpi", build_openpi, category="embodied", force=True)
