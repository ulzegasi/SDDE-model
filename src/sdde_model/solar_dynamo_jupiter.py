"""Jupiter-modulated companion to the original SDDE solar dynamo model.

This module reuses the established Julia session from :mod:`solar_dynamo` and
defines only additional, uniquely named Julia functions. The original bridge
and its public API remain unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import solar_dynamo as _original

JUPITER_ORBITAL_PERIOD_YEARS = 11.86

jl = None
_INITIALIZED = False


def _validate_theta(theta: Sequence[float]) -> tuple[float, ...]:
    theta = tuple(theta)
    if len(theta) != 7:
        raise ValueError(
            "The Jupiter model requires 7 parameters: "
            "(tau, T, Nd, sigma, Bmax, epsilon, phase)"
        )
    amplitude = theta[5]
    if not 0 <= amplitude < 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1")
    return theta


def _init_julia() -> None:
    """Initialize the original bridge, then add the Jupiter model once."""
    global _INITIALIZED, jl
    if _INITIALIZED:
        return

    _original._init_julia()
    jl = _original.jl
    jl.seval(
        r"""
        const JUPITER_ORBITAL_PERIOD_YEARS_ND = 11.86

        jupiter_nd_modulation(t, eps, phi) =
            1 + eps*cos(2*π*t/JUPITER_ORBITAL_PERIOD_YEARS_ND + phi)

        function f_jupiter_nd(u,h,p,t)
            τ, T, Nd, sigma, Bmax, eps, phi = p
            hist = h(p, t - T, idxs = 1)
            Nd_eff = Nd * jupiter_nd_modulation(t, eps, phi)
            du1 = u[2]
            du2 = -u[1]/τ^2 - 2*u[2]/τ - Nd_eff/τ^2*ftilde(hist, 1, Bmax)
            SA[du1, du2]
        end

        function g_jupiter_nd(u,h,p,t)
            τ, T, Nd, sigma, Bmax, eps, phi = p
            du1 = 0.0
            du2 = Bmax*sigma / (τ^(3/2))
            SA[du1, du2]
        end

        function bfield_jupiter_nd(θ, Tsim; dt=0.1, saveat=1.0, seed=nothing)
            @assert length(θ) == 7 "Jupiter model requires 7 parameters"
            τ, T, Nd, sigma, Bmax, eps, phi = θ
            @assert 0 <= eps < 1 "epsilon must satisfy 0 <= epsilon < 1"

            u0 = SA[Bmax, 0.0]
            h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)
            lags = (T,)
            tspan = (0.0, Tsim)

            prob = SDDEProblem(
                f_jupiter_nd,
                g_jupiter_nd,
                u0,
                h,
                tspan,
                θ;
                constant_lags=lags,
            )

            if seed !== nothing
                Random.seed!(seed)
            end

            solve(prob, EM(); dt=dt, saveat=saveat)
        end

        function sn_jupiter_nd(θ; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs
            sol = bfield_jupiter_nd(θ, Tsim; dt=dt, saveat=saveat, seed=seed)
            return map(abs2, sol[1, (Twarmup + 2):end])
        end

        function sn_batch_jupiter_nd(theta_batch; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seeds=nothing)
            n_batch = size(theta_batch, 1)
            out = Matrix{Float64}(undef, n_batch, Tobs)

            if seeds === nothing
                seeds = rand(1:2^31-1, n_batch)
            end
            @assert length(seeds) == n_batch "seeds must have one seed per theta row"

            @inbounds for i in 1:n_batch
                theta_i = tuple(theta_batch[i, :]...)
                out[i, :] .= sn_jupiter_nd(
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

        function sn_from_noise_jupiter_nd(theta, eps_dt; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            @assert dt > 0
            @assert length(theta) == 7 "Jupiter model requires 7 parameters"

            τ, T, Nd, sigma, Bmax, eps, phi = theta
            @assert 0 <= eps < 1 "epsilon must satisfy 0 <= epsilon < 1"
            Tsim = Twarmup + Tobs

            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"
            @assert length(eps_dt) >= Ndt "eps_dt too short: need Ndt = Tsim/dt"

            lag_steps = Int(round(T / dt))
            @assert lag_steps >= 1 "T/dt too small or dt too large"
            @assert abs(lag_steps*dt - T) < 1e-6 "T must be (approximately) a multiple of dt for this discretization"

            coeff = Bmax * sigma / (τ^(3/2))
            sdt = sqrt(dt)
            k = Int(round(1.0 / dt))
            @assert abs(k*dt - 1.0) < 1e-12 "dt must divide 1.0 when saveat==1.0"

            Nsave = Int(round(Tsim)) + 1
            @assert abs(Tsim - (Nsave - 1)) < 1e-9 "Tsim must be integer when saveat==1.0"

            B = Bmax
            dB = 0.0
            Bhist = fill(Bmax, lag_steps)
            hidx = 1

            y_save = Vector{Float64}(undef, Nsave)
            y_save[1] = B^2

            i = 0
            @inbounds for j in 1:(Nsave-1)
                for _sub in 1:k
                    i += 1
                    B_delay = Bhist[hidx]

                    t = (i - 1)*dt
                    Nd_eff = Nd * jupiter_nd_modulation(t, eps, phi)
                    du1 = dB
                    du2 = -B/τ^2 - 2*dB/τ - (Nd_eff/τ^2)*ftilde(B_delay, 1, Bmax)

                    dB_new = dB + du2*dt + coeff*sdt*eps_dt[i]
                    B_new = B + du1*dt

                    Bhist[hidx] = B_new
                    hidx += 1
                    if hidx > lag_steps
                        hidx = 1
                    end

                    B, dB = B_new, dB_new
                end
                y_save[j+1] = B^2
            end

            start = Twarmup + 2
            stop = start + Tobs - 1
            @assert stop <= length(y_save)
            return y_save[start:stop]
        end

        function sn_for_enca_jupiter_nd(theta; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs
            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"

            if seed !== nothing
                Random.seed!(seed)
            end
            eps_dt = randn(Ndt)
            return sn_from_noise_jupiter_nd(
                theta,
                eps_dt;
                Twarmup=Twarmup,
                Tobs=Tobs,
                dt=dt,
                saveat=saveat,
            )
        end

        function test_consistency_jupiter_nd(theta; seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0)
            Tsim = Twarmup + Tobs
            Ndt = Int(round(Tsim / dt))
            Random.seed!(seed)
            eps_dt = randn(Ndt)
            y1 = sn_from_noise_jupiter_nd(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
            y2 = sn_from_noise_jupiter_nd(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
            return maximum(abs.(y1 .- y2))
        end

        function sn_nrep_jupiter_nd(theta; nrep=8, seeds=nothing, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)
            if seeds === nothing
                seeds = rand(1:2^31-1, nrep)
            end

            L = Int(round(Tobs / saveat))
            X = Array{Float32}(undef, nrep, L)

            for j in 1:nrep
                y = sn_jupiter_nd(
                    theta;
                    Twarmup=Twarmup,
                    Tobs=Tobs,
                    dt=dt,
                    saveat=saveat,
                    seed=seeds[j],
                )
                X[j, :] .= Float32.(y)
            end
            return X
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
    theta = _validate_theta(theta)
    _init_julia()
    return jl.sn_jupiter_nd(
        theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed
    )


def sn_batch(
    theta_batch,
    Twarmup: int = 200,
    Tobs: int = 929,
    dt: float = 0.1,
    saveat: float = 1.0,
    seeds=None,
):
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    if theta_batch.ndim != 2 or theta_batch.shape[1] != 7:
        raise ValueError("theta_batch must have shape (n_batch, 7)")
    if np.any((theta_batch[:, 5] < 0) | (theta_batch[:, 5] >= 1)):
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1")

    _init_julia()
    kwargs = dict(Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
    if seeds is None:
        result = jl.sn_batch_jupiter_nd(theta_batch, **kwargs)
    else:
        seeds = np.asarray(seeds, dtype=np.int64)
        if seeds.ndim != 1 or seeds.shape[0] != theta_batch.shape[0]:
            raise ValueError("seeds must be 1D with length n_batch")
        result = jl.sn_batch_jupiter_nd(theta_batch, seeds=seeds, **kwargs)
    return np.asarray(result, dtype=np.float64)


def sn_for_enca(theta, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=None):
    theta = _validate_theta(theta)
    _init_julia()
    return jl.sn_for_enca_jupiter_nd(
        theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed
    )


def sn_from_noise(theta, eps, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    theta = _validate_theta(theta)
    _init_julia()
    return jl.sn_from_noise_jupiter_nd(
        theta, eps, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat
    )


def sn_nrep(theta, seeds, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    theta = _validate_theta(theta)
    _init_julia()
    result = jl.sn_nrep_jupiter_nd(
        theta,
        nrep=len(seeds),
        seeds=seeds,
        Twarmup=Twarmup,
        Tobs=Tobs,
        dt=dt,
        saveat=saveat,
    )
    return np.asarray(result, dtype=np.float32)


def test_consistency(theta, seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0):
    theta = _validate_theta(theta)
    _init_julia()
    return float(
        jl.test_consistency_jupiter_nd(
            theta,
            seed=seed,
            Twarmup=Twarmup,
            Tobs=Tobs,
            dt=dt,
            saveat=saveat,
        )
    )


hann_window = _original.hann_window
summary_statistics = _original.summary_statistics
summary_statistics_batch = _original.summary_statistics_batch
summary_statistics_ii = _original.summary_statistics_ii

__all__ = [
    "JUPITER_ORBITAL_PERIOD_YEARS",
    "hann_window",
    "sn",
    "sn_batch",
    "sn_for_enca",
    "sn_from_noise",
    "sn_nrep",
    "summary_statistics",
    "summary_statistics_batch",
    "summary_statistics_ii",
    "test_consistency",
]
