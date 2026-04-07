import nifty8 as ift
import resolve as rve
#from resolve.sky_model import _append_to_nonempty_string
#from resolve.sky_model import _parse_or_none

def _append_to_nonempty_string(s, append):
    if s == "":
        return s
    return s + append


def _parse_or_none(cfg, key, override={}, single_value=False):
    if single_value:
        if key in override:
            return override[key]
        if cfg[key] == "None":
            return None
        return cfg.getfloat(key)
    key0 = f"{key} mean"
    key1 = f"{key} stddev"
    if key in override:
        a, b = override[key]
        if a is None and b is None:
            return None
        return a, b
    if cfg[key0] == "None" and cfg[key1] == "None":
        return None
    if key0 in cfg:
        return (cfg.getfloat(key0), cfg.getfloat(key1))


def cfm_from_cfg(
    cfg,
    domain_dct,
    prefix,
    total_N=0,
    dofdex=None,
    override={},
    domain_prefix=None,
    matern=False
):
    assert len(prefix) > 0
    product_spectrum = len(domain_dct) > 1
    cfm = ift.CorrelatedFieldMaker(
        prefix if domain_prefix is None else domain_prefix,
        total_N=total_N,
    )
    for key_prefix, dom in domain_dct.items():
        ll = _append_to_nonempty_string(key_prefix, " ")
        if matern: 
            kwargs = {
                kk: _parse_or_none(cfg, f"{prefix} {ll}{kk}", override)
                for kk in ["scale", "loglogslope", "cutoff"]
            }
            cfm.add_fluctuations_matern(dom, **kwargs, prefix=key_prefix)
        else:
            kwargs = {
                kk: _parse_or_none(cfg, f"{prefix} {ll}{kk}", override)
                for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]
            }
            cfm.add_fluctuations(dom, **kwargs, prefix=key_prefix, dofdex=dofdex)

    foo = str(prefix)
    if not product_spectrum and len(key_prefix) != 0:
        foo += f" {key_prefix}"
    kwargs = {
        "offset_mean": _parse_or_none(
            cfg, f"{foo} zero mode offset", override=override, single_value=True
        ),
        "offset_std": _parse_or_none(cfg, f"{foo} zero mode", override=override),
    }
    cfm.set_amplitude_total_offset(**kwargs, dofdex=dofdex)
    return cfm
