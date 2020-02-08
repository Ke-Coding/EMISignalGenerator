from .scheduler import CosineLRScheduler as cos
from .scheduler import ConstScheduler as con

def get_sche(sc_cfg, optm):
    return globals()[sc_cfg.name](optimizer=optm, **sc_cfg.kwargs)
    if sc_cfg.sche == 'cos':
        return CosineLRScheduler(
            optimizer=optm,
            T_max=T_max,
            eta_min=sc_cfg.min_lr,
            base_lr=sc_cfg.base_lr,
            warmup_lr=sc_cfg.lr,
            warmup_steps=max(round(T_max * 0.04), 1),
            last_iter=start_epoch - 1
        )
    elif sc_cfg.sche == 'con':
        return ConstScheduler(
            lr=sc_cfg.lr
        )
    else:
        raise AttributeError(f'unknown scheduler type: {sc_cfg.sche}')