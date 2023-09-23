# Calculate V/Vmax for each galaxy using r-band absolute magnitude

import pandas as pd
import numpy as np
from scipy.optimize import brentq
from astropy.cosmology import Planck18
from astropy import units as u
import kcorrect.kcorrect
import multiprocessing as mp


def dist_mod(gal_z):
    return 5*np.log10(Planck18.luminosity_distance(gal_z).to(u.pc).value/10)

def cal_v_vmax(sdss_mags, sdss_mag_errs, gal_zs, survey_z_min, survey_z_max, m_max_r):
    sdss_korr = kcorrect.kcorrect.KcorrectSDSS(abcorrect=True, cosmo=Planck18)
    (maggies,ivars) = kcorrect.utils.sdss_asinh_to_maggies(mag=sdss_mags, mag_err=sdss_mag_errs)
    coeffs = sdss_korr.fit_coeffs_asinh(redshift = gal_zs, mag=sdss_mags, mag_err=sdss_mag_errs)
    ab_mags = sdss_korr.absmag(maggies=maggies, ivar=ivars, redshift=gal_zs, coeffs=coeffs)
    ab_mags_r = ab_mags[:,2]
    zmax_list = []
    for ab_mag_r, coeff, gal_z in zip(ab_mags_r, coeffs, gal_zs):
        if not np.isfinite(coeff).all():
            zmax_list.append(np.nan)
            continue
        else:
            def minimize_zmax(z):
                return m_max_r - dist_mod(z) - sdss_korr.kcorrect(redshift=z, coeffs=coeff)[2] - ab_mag_r
            
            if minimize_zmax(survey_z_max) >= 0:
                # 此时zmax为survey_z_max
                zmax_list.append(survey_z_max)
                continue
            elif minimize_zmax(survey_z_min) <= 0:
                # 此时出现了问题，理论上这个源不应该被观测到
                zmax_list.append(np.nan)
                continue
            else:
                # 此时zmax在survey_z_min和survey_z_max之间
                zmax_list.append(brentq(minimize_zmax, survey_z_min, survey_z_max))
    v_vmax = ((Planck18.comoving_volume(survey_z_max) - Planck18.comoving_volume(survey_z_min))\
        / (Planck18.comoving_volume(zmax_list) - Planck18.comoving_volume(survey_z_min))).value
    return v_vmax

def cal_v_vmax_mp(sdss_mags, sdss_mag_errs, gal_zs, z_min, z_max, m_max_r = 17.77 + 0.01):
    #
    print('Calculating V/Vmax...')
    v_vmax_list = []
    cpu_num = mp.cpu_count()
    with mp.Pool(processes=cpu_num) as p:
        v_vmax_list.append(
            p.starmap(cal_v_vmax, 
                zip(np.array_split((sdss_mags),cpu_num), 
                    np.array_split((sdss_mag_errs),cpu_num), 
                    np.array_split((gal_zs),cpu_num),
                    [z_min]*cpu_num,
                    [z_max]*cpu_num,
                    [m_max_r]*cpu_num)))
    return np.concatenate(v_vmax_list[0])