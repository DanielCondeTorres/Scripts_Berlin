import torch
from typing import Dict, Optional
from scipy.integrate import trapezoid,simps, quad
from scipy.optimize import curve_fit
import numpy as np

def harmonic(x, x0, k, V0=0):
    return k * (x - x0) ** 2 + V0
def harmonic_precise(x, x0, k_raw, V0=0): #Debe ir sin el cuadrado
    k = np.exp(k_raw)
    return k * (1 - np.cos(x - x0))  + V0

def harmonic_cosine_potential(x, x0, k_raw, V0=0):
    #k = np.exp(k_raw)
    #k = abs(k_raw)
    k=k_raw
    print('ESTA AQUI en COSINE POTENTIAL: ')
    return k * (np.cos(x) - np.cos(x0)) ** 2 + V0


def fit_harmonic_from_potential_estimates(
    bin_centers_nz: torch.Tensor, dG_nz: torch.Tensor, **kwargs
) -> Dict:
    r"""Method for fitting interaction parameters from data

    Parameters
    ----------
    bin_centers:
        Bin centers from a discrete histgram used to estimate the energy
        through logarithmic inversion of the associated Boltzmann factor
    dG_nz:
        The value of the energy :math:`U` as a function of the bin
        centers, as retrived via:

        ..math::

            U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}

        where :math:`\beta` is the inverse thermodynamic temperature and
        :math:`p(x)` is the normalized probability distribution of
        :math:`x`.

    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """

    # remove noise by discarding signals
    integral = torch.tensor(
        float(simps(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )

    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral) #1e-4
    try:
        popt, _ = curve_fit(
            harmonic,
            bin_centers_nz[mask],
            dG_nz[mask],
            p0=[bin_centers_nz[torch.argmin(dG_nz[mask])], 100, -2],# 60 y -1
        )
        stat = {"k": popt[1], "x_0": popt[0]}
        print('STATS: ',stat)
    except:
        print(f"failed to fit potential estimate for harmonic")
        stat = {
            "k": torch.tensor(float("nan")),
            "x_0": torch.tensor(float("nan")),
        }
    return stat


def fit_harmonic_from_potential_estimates_angle(
    bin_centers_nz: torch.Tensor, dG_nz: torch.Tensor, **kwargs
) -> Dict:
    r"""Method for fitting interaction parameters from data

    Parameters
    ----------
    bin_centers:
        Bin centers from a discrete histgram used to estimate the energy
        through logarithmic inversion of the associated Boltzmann factor
    dG_nz:
        The value of the energy :math:`U` as a function of the bin
        centers, as retrived via:

        ..math::

            U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}

        where :math:`\beta` is the inverse thermodynamic temperature and
        :math:`p(x)` is the normalized probability distribution of
        :math:`x`.

    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """

    # remove noise by discarding signals
    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )

    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
    try:
        popt, _ = curve_fit(
            harmonic_cosine_potential,
            bin_centers_nz[mask],
            dG_nz[mask],
            p0=[bin_centers_nz[torch.argmin(dG_nz[mask])], 60, -3],# 60 y -1
        )
        stat = {"k": popt[1], "x_0": popt[0]}
        print('STATS: ',stat)
    except:
        print(f"failed to fit potential estimate for harmonic")
        stat = {
            "k": torch.tensor(float("nan")),
            "x_0": torch.tensor(float("nan")),
        }
    print('ESTIMANDO ÁNGULOS EN EL DEL ANGULO: ')
    return stat

