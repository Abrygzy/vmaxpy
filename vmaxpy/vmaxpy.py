# Calculate V/Vmax for each galaxy using r-band absolute magnitude

import pandas as pd
import numpy as np
from scipy.optimize import brentq
from astropy.cosmology import Planck18
from astropy import units as u
import kcorrect.kcorrect
import multiprocessing as mp


def dist_mod(gal_z, cosmo):
    return 5*np.log10(cosmo.luminosity_distance(gal_z).to(u.pc).value/10)

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

def cal_v_vmax_ext(sdss_mags, sdss_mag_errs, sdss_exts, gal_zs, survey_z_min, survey_z_max,cosmo, m_max_r):
    sdss_korr = kcorrect.kcorrect.KcorrectSDSS(abcorrect=True, cosmo=cosmo)
    (maggies,ivars) = kcorrect.utils.sdss_asinh_to_maggies(mag=sdss_mags, mag_err=sdss_mag_errs,extinction=sdss_exts)
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
                return m_max_r - dist_mod(z, cosmo) - sdss_korr.kcorrect(redshift=z, coeffs=coeff)[2] - ab_mag_r
            
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
    # v_vmax = ((cosmo.comoving_volume(survey_z_max) - cosmo.comoving_volume(survey_z_min))\
    #     / (cosmo.comoving_volume(zmax_list) - cosmo.comoving_volume(survey_z_min))).value
    vmax = (cosmo.comoving_volume(zmax_list) - cosmo.comoving_volume(survey_z_min)).value
    return vmax

def cal_v_vmax_ext_mp(sdss_mags, sdss_mag_errs,sdss_exts, gal_zs, z_min, z_max, cosmo, m_max_r = 17.77 + 0.01):
    #
    print('Calculating V/Vmax...')
    v_vmax_list = []
    cpu_num = mp.cpu_count()
    with mp.Pool(processes=cpu_num) as p:
        v_vmax_list.append(
            p.starmap(cal_v_vmax_ext, 
                zip(np.array_split((sdss_mags),cpu_num), 
                    np.array_split((sdss_mag_errs),cpu_num), 
                    np.array_split((sdss_exts),cpu_num), 
                    np.array_split((gal_zs),cpu_num),
                    [z_min]*cpu_num,
                    [z_max]*cpu_num,
                    [cosmo]*cpu_num,
                    [m_max_r]*cpu_num)))
    return np.concatenate(v_vmax_list[0])

def cal_zmax_sdss(sdss_mags, sdss_mag_errs, sdss_exts, gal_zs, cosmo,survey_z_min, survey_z_max, m_max_rs):
    '''
    Calculate zmax for each galaxy using r-band absolute magnitude, for SDSS galaxies especially
    Input:
        sdss_mags: magnitude in each band, directly from SDSS database (model, petro, etc.) np.array (n_gal, n_band=5)
        sdss_mag_errs: magnitude error in each band, directly from SDSS database (model, petro, etc.) np.array (n_gal, n_band=5)
        sdss_exts: extinction in each band, directly from SDSS database (model, petro, etc.) np.array (n_gal, n_band=5)
        gal_zs: redshift of each galaxy np.array (n_gal)
        cosmo: cosmology, astropy.cosmology object
        survey_z_min: minimum redshift of the survey, float
        survey_z_max: maximum redshift of the survey, float
        m_max_rs: maximum magnitude in r band, np.array (n_gal)
    '''

    # 先判断星系的红移是否在survey_z_min和survey_z_max之间, 如果不在, 直接报错
    if not ((gal_zs >= survey_z_min) & (gal_zs <= survey_z_max)).all():
        raise ValueError('Some galaxies are not in the redshift range of the survey!')

    # 先利用kcorrect, 得到每一个星系的绝对星等, 以及这个星系的 kcorrect template coefficients
    kc = kcorrect.kcorrect.KcorrectSDSS(abcorrect=True, cosmo=cosmo, redshift_range=[0, survey_z_max+0.1]);
    coeffs = kc.fit_coeffs_asinh(redshift = gal_zs, mag=sdss_mags, mag_err=sdss_mag_errs, extinction=sdss_exts)
    (maggies,ivars) = kcorrect.utils.sdss_asinh_to_maggies(mag=sdss_mags, mag_err=sdss_mag_errs,extinction=sdss_exts)
    ab_mags = kc.absmag(maggies=maggies, ivar=ivars, redshift=gal_zs, coeffs=coeffs)
    ab_mags_r = ab_mags[:,2]
    
    # 对每个星系计算zmax
    zmax_list = [] 
    for ab_mag_r, coeff, m_max_r in zip(ab_mags_r, coeffs, m_max_rs):
        # 对于 kcorrect 无法处理的星系 (coeff 为 nan), 直接返回nan
        if not np.isfinite(coeff).all():
            zmax_list.append(np.nan)
            continue
        else:
            # 定义函数 minimize_zmax, 当该函数取值为0时, 对应的z即为zmax
            # 此函数随着z的增大而单调减小, 因此可以用 brentq 方法求解
            def minimize_zmax(z):
                return m_max_r - cosmo.distmod(z).value - kc.kcorrect(redshift=z, coeffs=coeff)[2] - ab_mag_r
            # 如果在survey_z_max处函数值大于0, 则zmax为survey_z_max, 表示该星系在survey_z_max处仍然可见
            if minimize_zmax(survey_z_max) >= 0:
                zmax_list.append(survey_z_max)
                continue
            # 如果在survey_z_min处函数值小于0, 则出现了问题, 理论上这个源在survey_z_min处都看不到, 返回nan
            elif minimize_zmax(survey_z_min) <= 0:
                zmax_list.append(np.nan)
                continue
            # 否则, 星系的zmax在survey_z_min和survey_z_max之间, 用 brentq 方法求解
            else:
                zmax_list.append(brentq(minimize_zmax, survey_z_min, survey_z_max))
    return np.array(zmax_list)

def cal_zmax_sdss_mp(sdss_mags, sdss_mag_errs, sdss_exts, gal_zs, cosmo,survey_z_min, survey_z_max, m_max_rs):
    print('Calculating z_max...')
    zmax_list = []
    cpu_num = mp.cpu_count()
    with mp.Pool(processes=cpu_num) as p:
        zmax_list.append(
            p.starmap(cal_zmax_sdss, 
                zip(np.array_split((sdss_mags),cpu_num), 
                    np.array_split((sdss_mag_errs),cpu_num), 
                    np.array_split((sdss_exts),cpu_num), 
                    np.array_split((gal_zs),cpu_num),
                    [cosmo]*cpu_num,
                    [survey_z_min]*cpu_num,
                    [survey_z_max]*cpu_num,
                    np.array_split((m_max_rs),cpu_num))))
    return np.concatenate(zmax_list[0])