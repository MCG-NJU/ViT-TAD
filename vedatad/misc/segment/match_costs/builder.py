from vedacore.misc import build_from_cfg, registry

def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, registry, 'match_cost', default_args)
