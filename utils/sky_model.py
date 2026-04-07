import nifty8 as ift
import numpy as np

from resolve.constants import str2rad
from resolve.polarization_space import PolarizationSpace
from resolve.sky_model import default_sky_domain
from resolve.util import assert_sky_domain
from resolve.util import my_asserteq, my_assert
from .ift_cfm_maker import cfm_from_cfg
from .utilities import mas_to_rad




def sky_model_diffuse(cfg, img_size, fov, zeromode_offset=None, source_number=0, matern=False):
    """
    Construct and return a diffuse sky model based on the provided configuration.
    
    :param cfg: loaded config object
    :param img_size: tuple of image size (nx, ny)
    :param fov: tuple of field of view (fov_x, fov_y) in mas
    :param zeromode_offset: override zeromode offset value
    :param source_number: source number (0 for main source, >0 for additional sources)
    :param matern: whether to use matern correlation kernal (with a cutoff) for the sky model
    """
    sdom = _spatial_dom(img_size, fov)
    pdom = PolarizationSpace(cfg["polarization"].split(","))
    my_asserteq(len(pdom.labels), 1)
    pol_label = f"{pdom.labels[0]}"

    additional = {}


    if cfg["freq mode"] == "single":
        op, aa = _single_freq_logsky(cfg, img_size, fov, pol_label, source_number, zeromode_offset=zeromode_offset, matern=matern)
    else:
        err = f"multi freq mode is not implemented here. "
        raise NotImplementedError(err)

    logsky = op
    additional = {**additional, **aa}

    tgt = default_sky_domain(pdom=pdom, sdom=sdom)

    sky = ift.exp(logsky).ducktape_left(tgt)
    assert_sky_domain(sky.target)

    return sky, additional



def _single_freq_logsky(cfg, imsize, fov, pol_label, source_number=0, zeromode_offset=None, matern=False):
    sdom = _spatial_dom(imsize, fov)
    my_assert(type(source_number) == int)
    if source_number==0:
        override = {"stokesI diffuse space i0 zero mode offset": zeromode_offset}
        cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse", override=override, matern=matern)
    if source_number > 0:
        override = {"stokesI diffuse space i0 zero mode offset": zeromode_offset}
        cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse", domain_prefix=f"source{source_number} ", override=override, matern=matern)
    op = cfm.finalize(0)
    additional = {
        f"logdiffuse stokes{pol_label} power spectrum": cfm.power_spectrum,
        f"logdiffuse stokes{pol_label}": op,
    }

    return op, additional


def _spatial_dom(imsize, fov):
    nx, ny = imsize
    fov_x, fov_y = fov
    dx = mas_to_rad(fov_x) / nx
    dy = mas_to_rad(fov_y) / ny
    return ift.RGSpace([nx, ny], [dx, dy])