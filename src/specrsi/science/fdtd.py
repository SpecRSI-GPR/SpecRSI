"""Actual 2-D TM Yee Maxwell solver with split-field absorbing layers.

Normalized units c=epsilon0=mu0=1. Ez is cell centered, Hx/Hy staggered
in space and half a timestep. Flux interpolates both time and space.
Candidate geometry and runtime grid settings define the material discretization.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
from numba import njit

from specrsi.benchmarks.science import fdtd_score
from .common import InvalidCandidate, exact_keys, scalar


def design(candidate: dict, c: dict) -> np.ndarray:
    exact_keys(candidate, ("split_fraction", "bend_exponent", "upper_width_scale", "lower_width_scale", "grating_period_um", "grating_depth"))
    bounds = {"split_fraction": (0.15, 0.85), "bend_exponent": (0.5, 3), "upper_width_scale": (0.6, 1.4),
              "lower_width_scale": (0.6, 1.4), "grating_period_um": (0.2, 1.2), "grating_depth": (0, 0.8)}
    for key, bound in bounds.items():
        scalar(candidate[key], *bound, key)
    nx, ny, dx = c["nx"], c["ny"], c["cell_size_um"]
    y = (np.arange(ny) - (ny-1)/2) * dx
    epsilon = np.full((nx, ny), c["epsilon_cladding"], dtype=float)
    start, stop = [int(v * nx) for v in c["design_x_fractions"]]
    junction = start + (stop-start) * candidate["split_fraction"] * 0.5
    for i in range(nx):
        if i < junction:
            core = np.abs(y) <= c["input_width_um"]/2
        else:
            progress = min(1., (i-junction)/(stop-junction))
            offset = c["output_offset_um"] * progress**candidate["bend_exponent"]
            upper_scale = 1 + (candidate["upper_width_scale"]-1) * math.sin(math.pi*progress)
            lower_scale = 1 + (candidate["lower_width_scale"]-1) * math.sin(math.pi*progress)
            width = c["input_width_um"] + (c["output_width_um"]-c["input_width_um"]) * progress
            upper = np.abs(y-offset) <= width*upper_scale/2
            lower = np.abs(y+offset) <= width*lower_scale/2
            core = upper | lower
            if junction < i < stop and math.sin(2*math.pi*i*dx/candidate["grating_period_um"]) > 0:
                epsilon[i, upper] = c["epsilon_core"] * (1-candidate["grating_depth"])
        untouched = core & (epsilon[i] == c["epsilon_cladding"])
        epsilon[i, untouched] = c["epsilon_core"]
    if epsilon.shape != (nx,ny) or not np.all(np.isfinite(epsilon)) or epsilon.min() < 1:
        raise InvalidCandidate("Invalid dielectric field")
    return epsilon


@njit(cache=True)
def _yee(epsilon, wavelength_cells, steps, courant, absorber, strength, source_x, monitor_x, source,
         apertures, settle_step, ramp):
    nx, ny = epsilon.shape
    ez = np.zeros((nx,ny)); ex = ez.copy(); ey = ez.copy(); hx = ez.copy(); hy = ez.copy()
    ax = np.ones(nx); bx = np.full(nx,courant)
    ay = np.ones(ny); by = np.full(ny,courant)
    ahx = np.ones(nx); bhx = np.full(nx,courant)
    ahy = np.ones(ny); bhy = np.full(ny,courant)
    for i in range(nx):
        distance = min(i,nx-1-i)
        sigma = strength * max(0., (absorber-distance)/absorber)**3
        ax[i] = (1-sigma/2)/(1+sigma/2); bx[i] /= 1+sigma/2
        distance = min(i+0.5,nx-1-(i+0.5))
        sigma = strength * max(0., (absorber-distance)/absorber)**3
        ahx[i] = (1-sigma/2)/(1+sigma/2); bhx[i] /= 1+sigma/2
    for j in range(ny):
        distance = min(j,ny-1-j)
        sigma = strength * max(0., (absorber-distance)/absorber)**3
        ay[j] = (1-sigma/2)/(1+sigma/2); by[j] /= 1+sigma/2
        distance = min(j+0.5,ny-1-(j+0.5))
        sigma = strength * max(0., (absorber-distance)/absorber)**3
        ahy[j] = (1-sigma/2)/(1+sigma/2); bhy[j] /= 1+sigma/2
    flux = np.zeros((2,2))
    peak = 0.
    split = (steps+settle_step)//2
    for t in range(steps):
        for i in range(nx-1):
            for j in range(ny-1):
                hx[i,j] = ahy[j]*hx[i,j] - bhy[j]*(ez[i,j+1]-ez[i,j])
                hy[i,j] = ahx[i]*hy[i,j] + bhx[i]*(ez[i+1,j]-ez[i,j])
        amp = math.sin(2*math.pi*courant*t/wavelength_cells) * min(1., t/ramp)**2
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                old = ez[i,j]
                ex[i,j] = ax[i]*ex[i,j] + bx[i]*(hy[i,j]-hy[i-1,j])/epsilon[i,j]
                ey[i,j] = ay[j]*ey[i,j] - by[j]*(hx[i,j]-hx[i,j-1])/epsilon[i,j]
                if i == source_x:
                    ex[i,j] += amp*source[j]/2
                    ey[i,j] += amp*source[j]/2
                ez[i,j] = ex[i,j]+ey[i,j]
                if i == monitor_x and t >= settle_step:
                    poynting = -(ez[i,j]+old)*0.25*(hy[i,j]+hy[i-1,j])
                    window = 0 if t < split else 1
                    for port in range(2):
                        if apertures[port,j]:
                            flux[port,window] += poynting
                if abs(ez[i,j]) > peak:
                    peak = abs(ez[i,j])
        if not math.isfinite(peak) or peak > 1e6:
            return flux, peak
    flux[:,0] /= max(1,split-settle_step)
    flux[:,1] /= max(1,steps-split)
    return flux, peak


def solve_field(epsilon: np.ndarray, c: dict, wavelength_um: float, reference: bool=False) -> dict:
    nx, ny, dx = c["nx"],c["ny"],c["cell_size_um"]
    y = (np.arange(ny)-(ny-1)/2)*dx
    source = np.exp(-0.5*(y/(c["input_width_um"]/2))**2)
    apertures = np.zeros((2,ny),dtype=np.bool_)
    for port, center in enumerate(([0,0] if reference else [c["output_offset_um"],-c["output_offset_um"]])):
        apertures[port] = np.abs(y-center) <= c["output_width_um"]*1.4
    flux, peak = _yee(epsilon, wavelength_um/dx, c["steps"], c["courant"], c["absorber_cells"],
                      c["absorber_strength"], int(c["source_x_fraction"]*nx),int(c["monitor_x_fraction"]*nx),
                      source, apertures, int(c["steps"]*c["settle_fraction"]),c["source_ramp_steps"])
    if not np.all(np.isfinite(flux)) or not math.isfinite(peak) or peak > 1e6 or np.min(flux) < -1e-7:
        raise InvalidCandidate("FDTD divergence/nonphysical outgoing flux")
    mean = np.maximum(0,flux.mean(axis=1))
    drift = np.abs(flux[:,1]-flux[:,0])/np.maximum(mean,1e-10)
    if np.any((drift>c["max_late_window_relative_drift"]) & (mean>1e-5)):
        raise InvalidCandidate("FDTD outgoing flux did not settle within the registered late-window tolerance")
    return {"flux": mean.tolist(), "late_window_relative_drift": drift.tolist(), "peak_field": float(peak)}


def simulate(candidate: dict, c: dict) -> dict:
    if c["courant"] >= 1/math.sqrt(2):
        raise RuntimeError("Registered FDTD Courant factor unstable")
    epsilon = design(candidate,c)
    reference = np.full(epsilon.shape,c["epsilon_cladding"])
    y = (np.arange(c["ny"])-(c["ny"]-1)/2)*c["cell_size_um"]
    reference[:,np.abs(y)<=c["input_width_um"]/2] = c["epsilon_core"]
    rows, transmissions = [], {}
    for k,wavelength in enumerate(c["wavelengths_um"]):
        incident = solve_field(reference,c,wavelength,True)
        measured = solve_field(epsilon,c,wavelength)
        normalization = incident["flux"][0]
        if normalization <= 1e-8:
            raise InvalidCandidate("Reference incident flux is zero")
        upper,lower = [x/normalization for x in measured["flux"]]
        transmissions.update({f"T{k+1}A":upper,f"T{k+1}B":lower})
        rows.append({"wavelength_um":wavelength,"incident":incident,"design":measured})
    result = fdtd_score(transmissions,valid=True)
    return {"native_score":result.score,"native_unit":result.metrics["native_unit"],"transmissions":transmissions,
            "wavelength_runs":rows,"epsilon_sha256":hashlib.sha256(epsilon.tobytes()).hexdigest(),
            "grid":[c["nx"],c["ny"]],"steps_per_solve":c["steps"],"actual_field_solves":4,
            "simulator":"TM Yee FDTD with split-field polynomial absorbing layer; straight-guide flux reference"}
