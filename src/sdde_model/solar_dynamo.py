"""
Module name: sdde_solar_dynamo_julia.py
Author: Simone Ulzega, March 2026, simone.ulzega@zhaw.ch 
- src.sdde_solar_dynamo_julia provides a Python interface to the Julia SDDE solver:
- load packages and defines Julia model functions
- Julia functions are initialized when _init_julia() is called
- _init_julia() will be called lazily inside sn() and summary_statistics() when they are first called.
- _init_julia is governed by a global _INITIALIZED flag to ensure it only runs once.
- sn() runs the SDDE solver and returns the time series of the magnetic field strength
- summary_statistics() computes summary statistics from the time series using FFT
- We do the Julia setup lazily, only when sn() or summary_statistics() is called
- For stability, call sdde_model.init_julia() at the very top of your main script
"""
  
from __future__ import annotations
from typing import Iterable, Optional, Sequence
import numpy as np

# IMPORTANT:
# - Do NOT import juliacall.Main at module import time (can crash depending on import order).
# - We only grab it inside _init_julia(), ideally after sdde_model.init_julia() was called.

jl = None
_INITIALIZED = False

def _init_julia():
    """
    One-time Julia setup: imports packages and defines Julia functions.

    For stability:
    - Call sdde_model.init_julia() at the very top of your main script
      BEFORE importing tensorflow or other native-library-heavy modules.
    """
    global _INITIALIZED, jl
    if _INITIALIZED:
        return

    # Bootstrap Julia lazily if the caller did not do it explicitly first.
    if jl is None:
        from .bootstrap import init_julia

        jl = init_julia()

    # Julia imports (one-time)
    jl.seval("using StochasticDelayDiffEq")
    jl.seval("using SpecialFunctions: erf")
    jl.seval("using StaticArrays")
    jl.seval("using FFTW")
    jl.seval("using Random")
    jl.seval("using DiffEqNoiseProcess: NoiseGrid")

    # Define Julia functions (one-time)
    jl.seval(
        r"""
        ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))

        function f(u,h,p,t)
            τ, T, Nd, sigma, Bmax = p
            hist = h(p, t - T, idxs = 1)
            du1 = u[2]
            du2 = -u[1]/τ^2 - 2*u[2]/τ - Nd/τ^2*ftilde(hist, 1, Bmax)
            SA[du1, du2]
        end

        function g(u,h,p,t)
            τ, T, Nd, sigma, Bmax = p
            du1 = 0.0
            du2 = Bmax*sigma / (τ^(3/2))
            SA[du1, du2]
        end

        function bfield(θ, Tsim; dt=0.1, saveat=1.0, seed=nothing, noise=nothing)
            τ, T, Nd, sigma, Bmax = θ
            u0 = SA[Bmax, 0.0]
            h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)
            lags = (T,)
            tspan = (0.0, Tsim)

            prob = SDDEProblem(
                f,
                g,
                u0,
                h,
                tspan,
                θ;
                constant_lags=lags,
                noise=noise,
            )

            if seed !== nothing
                Random.seed!(seed)
            end

            solve(prob, EM(); dt=dt, saveat=saveat)
        end

        function sn(θ; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs
            sol = bfield(θ, Tsim; dt=dt, saveat=saveat, seed=seed)
            y = map(abs2, sol[1, (Twarmup + 2):end])
            return y
        end

        function sn_batch(theta_batch; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seeds=nothing)
            n_batch = size(theta_batch, 1)
            out = Matrix{Float64}(undef, n_batch, Tobs)

            if seeds === nothing
                seeds = rand(1:2^31-1, n_batch)
            end
            @assert length(seeds) == n_batch "seeds must have one seed per theta row"

            @inbounds for i in 1:n_batch
                theta_i = tuple(theta_batch[i, :]...)
                out[i, :] .= sn(
                    theta_i;
                    Twarmup=Twarmup,
                    Tobs=Tobs,
                    dt=dt,
                    saveat=saveat,
                    seed=seeds[i],
                )
            end
            return out
        end
        
        # Deterministic SDDEProblem/EM solve given dt-level bare Gaussian
        # increments. NoiseGrid exposes the noise without replacing the
        # solver's continuous-delay history handling.
        function sn_from_noise(theta, eps_dt; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            @assert dt > 0
            @assert length(theta) == 5 "Original model requires 5 parameters"
            Tsim = Twarmup + Tobs
            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"
            @assert length(eps_dt) >= Ndt "eps_dt too short: need Ndt = Tsim/dt"

            noise_times = collect(range(0.0; step=dt, length=Ndt + 1))
            brownian_path = Vector{Float64}(undef, Ndt + 1)
            brownian_path[1] = 0.0
            sqrt_dt = sqrt(dt)
            @inbounds for i in 1:Ndt
                brownian_path[i + 1] = brownian_path[i] + sqrt_dt*eps_dt[i]
            end
            noise = NoiseGrid(noise_times, brownian_path)

            sol = bfield(
                theta,
                Tsim;
                dt=dt,
                saveat=saveat,
                noise=noise,
            )
            return map(abs2, sol[1, (Twarmup + 2):end])
        end

        function sn_from_noise_batch(
            theta_batch,
            eps_batch;
            Twarmup=200,
            Tobs=929,
            dt=0.1,
            saveat=1.0,
        )
            n_batch = size(theta_batch, 1)
            @assert size(theta_batch, 2) == 5 "theta_batch must have 5 columns"
            @assert size(eps_batch, 1) == n_batch "eps_batch must have one row per theta"
            out = Matrix{Float64}(undef, n_batch, Tobs)

            @inbounds for i in 1:n_batch
                theta_i = tuple(theta_batch[i, :]...)
                out[i, :] .= sn_from_noise(
                    theta_i,
                    view(eps_batch, i, :);
                    Twarmup=Twarmup,
                    Tobs=Tobs,
                    dt=dt,
                    saveat=saveat,
                )
            end
            return out
        end

        function sn_for_enca(theta; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs

            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"

            if seed !== nothing
                Random.seed!(seed)
            end
            eps_dt = randn(Ndt)
            return sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
        end

        function test_consistency(theta; seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs

            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"

            Random.seed!(seed)
            eps_dt = randn(Ndt)

            y1 = sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
            y2 = sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)

            return maximum(abs.(y1 .- y2))
        end
        
        # ------------------------------------------------------------
        # This is the INCA-friendly model
        # ------------------------------------------------------------
        function sn_nrep(theta; nrep=8, seeds=nothing, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)

            if seeds === nothing
                seeds = rand(1:2^31-1, nrep)
            end

            L = Int(round(Tobs / saveat))

            X = Array{Float32}(undef, nrep, L)

            for j in 1:nrep
                y = sn(theta;
                    Twarmup=Twarmup,
                    Tobs=Tobs,
                    dt=dt,
                    saveat=saveat,
                    seed=seeds[j])

                X[j, :] .= Float32.(y)
            end

            return X
        end

        hann_window(Tmax) = [0.5*(1 - cos(2.0*π*(t-1)/(Tmax-1))) for t in 1:Tmax]

        function summary_statistics(data, window=hann_window(length(data)); fourier_range=1:6:120)
            fs = FFTW.ifft(window .* data)
            ss = abs.(fs[fourier_range])
            return ss
        end

        function summary_statistics_batch(data_batch, window=hann_window(size(data_batch, 2)); fourier_range=1:6:120)
            n_batch = size(data_batch, 1)
            n_stats = length(fourier_range)
            out = Matrix{Float64}(undef, n_batch, n_stats)

            @inbounds for i in 1:n_batch
                out[i, :] .= summary_statistics(vec(data_batch[i, :]), window; fourier_range=fourier_range)
            end
            return out
        end

        function summary_statistics_ii(data, window=hann_window(length(data)); fourier_range=1:6:120)
            fs = FFTW.ifft(window .* data)[1:120]
            ss = [real.(fs); imag.(fs)][fourier_range]
            return ss
        end
        """
    )

    _INITIALIZED = True


def sn(
    theta: Sequence[float],
    Twarmup: int = 200,
    Tobs: int = 929,
    dt: float = 0.1,
    saveat: float = 1.0,
    seed: Optional[int] = None,
):
    _init_julia()
    # juliacall likes tuples for small fixed-size vectors
    return jl.sn(tuple(theta), Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed)

def sn_batch(
    theta_batch,
    Twarmup: int = 200,
    Tobs: int = 929,
    dt: float = 0.1,
    saveat: float = 1.0,
    seeds=None,
):
    _init_julia()
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    if theta_batch.ndim != 2:
        raise ValueError("theta_batch must be 2D (n_batch, n_para)")

    if seeds is None:
        X = jl.sn_batch(theta_batch, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
    else:
        seeds = np.asarray(seeds, dtype=np.int64)
        if seeds.ndim != 1 or seeds.shape[0] != theta_batch.shape[0]:
            raise ValueError("seeds must be 1D with length n_batch")
        X = jl.sn_batch(theta_batch, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seeds=seeds)

    return np.asarray(X, dtype=np.float64)


def hann_window(Tmax: int):
    _init_julia()
    return jl.hann_window(Tmax)


def summary_statistics(
    data,
    window=None,
    fourier_range=None,
):
    _init_julia()
    if window is None and fourier_range is None:
        return jl.summary_statistics(data)
    if fourier_range is None:
        return jl.summary_statistics(data, window)
    return jl.summary_statistics(data, window, fourier_range=fourier_range)


def summary_statistics_batch(
    data_batch,
    window=None,
    fourier_range=None,
):
    _init_julia()
    data_batch = np.asarray(data_batch, dtype=np.float64)
    if data_batch.ndim != 2:
        raise ValueError("data_batch must be 2D (n_batch, n_samples)")

    if window is None and fourier_range is None:
        return np.asarray(jl.summary_statistics_batch(data_batch), dtype=np.float64)
    if fourier_range is None:
        return np.asarray(jl.summary_statistics_batch(data_batch, window), dtype=np.float64)
    return np.asarray(
        jl.summary_statistics_batch(data_batch, window, fourier_range=fourier_range),
        dtype=np.float64,
    )


def summary_statistics_ii(
    data,
    window=None,
    fourier_range=None,
):
    _init_julia()
    if window is None and fourier_range is None:
        return jl.summary_statistics_ii(data)
    if fourier_range is None:
        return jl.summary_statistics_ii(data, window)
    return jl.summary_statistics_ii(data, window, fourier_range=fourier_range)

def sn_for_enca(theta, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=None):
    _init_julia()
    return jl.sn_for_enca(tuple(theta), Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed)

def sn_from_noise(theta, eps, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    theta = tuple(theta)
    if len(theta) != 5:
        raise ValueError("Original model requires 5 parameters")
    _init_julia()
    return jl.sn_from_noise(theta, eps, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)


def sn_from_noise_batch(
    theta_batch,
    eps_batch,
    Twarmup=200,
    Tobs=929,
    dt=0.1,
    saveat=1.0,
):
    """Simulate an original-model batch with explicit Gaussian EM increments."""
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    eps_batch = np.asarray(eps_batch, dtype=np.float64)
    if theta_batch.ndim != 2 or theta_batch.shape[1] != 5:
        raise ValueError("theta_batch must have shape (n_batch, 5)")
    expected_increments = int(round((Twarmup + Tobs) / dt))
    expected_shape = (theta_batch.shape[0], expected_increments)
    if eps_batch.ndim != 2 or eps_batch.shape != expected_shape:
        raise ValueError(f"eps_batch must have shape {expected_shape}")

    _init_julia()
    result = jl.sn_from_noise_batch(
        theta_batch,
        eps_batch,
        Twarmup=Twarmup,
        Tobs=Tobs,
        dt=dt,
        saveat=saveat,
    )
    return np.asarray(result, dtype=np.float64)

def sn_nrep(theta, seeds, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    _init_julia()
    # juliacall likes tuples for small fixed-size vectors
    X = jl.sn_nrep(
        tuple(theta),
        nrep=len(seeds),
        seeds=seeds,
        Twarmup=Twarmup,
        Tobs=Tobs,
        dt=dt,
        saveat=saveat,
    )
    return np.asarray(X, dtype=np.float32)

def test_consistency(theta, seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0):
    _init_julia()
    return float(jl.test_consistency(tuple(theta), seed=seed, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat))
