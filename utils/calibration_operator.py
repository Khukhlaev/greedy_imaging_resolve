import resolve as rve
import nifty8 as ift
from .ift_cfm_maker import cfm_from_cfg
import ducc0
import numpy as np


def dofdex_or_none(cfg, key, total_N, single_gain = False):
    if cfg[key] == "False":
        return None
    if cfg[key] == "True":
        if single_gain:
            return np.arange(total_N) # In the case of one polarization mode, one kernel per antenna
        return np.hstack([np.arange(total_N//2), np.arange(total_N//2)]) # One kernel per antenna, same for LL and RR
    else:
        return np.fromstring(cfg[key], dtype=int, sep=',')



def get_calibration_operator(cfg, obs, tmin, tmax):
    sol_int = cfg["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2
    uantennas = rve.unique_antennas(obs)

    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / sol_int)), sol_int
    )

    total_N = obs.vis.val.shape[0] * len(uantennas) # 1 or 2 pol modes * # antennas 
    single_gain = obs.vis.val.shape[0] == 1

    dofdex_phase = dofdex_or_none(cfg["gain_phase"], f"diff_correlation_kernels", total_N, single_gain)
    dofdex_logamp = dofdex_or_none(cfg["gain_logamplitude"], f"diff_correlation_kernels", total_N, single_gain)

    dd = {"time": time_domain}
    cfm_kwargs_phase = {"total_N": total_N, "dofdex":dofdex_phase}
    cfm_kwargs_logamp = {"total_N": total_N, "dofdex":dofdex_logamp}
    phase_ = cfm_from_cfg(
        cfg["gain_phase"],
        dd,
        "gain_phase",
        domain_prefix="gain_phase",
        **cfm_kwargs_phase,
    ).finalize(0)

    uncorrelated_gain_phase = cfg["gain_phase"]["uncorrelated_gain_phase"]
    phase_amp = cfg["gain_phase"].getfloat("uncorrelated_gain_phase_amp")

    if uncorrelated_gain_phase == "True":
        phase_ = phase_amp * np.pi * ift.FieldAdapter(phase_.target, "gain_phase")  
        
    logamp_ = cfm_from_cfg(
        cfg["gain_logamplitude"],
        dd,
        "gain_logamplitude",
        domain_prefix="gain_logamplitude",
        **cfm_kwargs_logamp,
    ).finalize(0)
    
    # if cfg["gain_logamplitude"]["uncorrelated_gain_logamplitude"] == "True":
    #     logamp_ = cfg["gain_logamplitude"].getfloat("uncorrelated_gain_logamplitude_amp") * ift.FieldAdapter(logamp_.target, "gain_logamplitude") 

    pdom, _, fdom = obs.vis.domain
    reshaper = ift.DomainChangerAndReshaper(
        phase_.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    phase = reshaper @ phase_
    logamp = reshaper @ logamp_

    return antenna_dct, rve.calibration_distribution(obs, phase, logamp, antenna_dct), phase, logamp