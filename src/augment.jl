using IntervalSets
using Statistics

export rand_timesampleshift
export rand_polarityinversion
export rand_tanhdistortion
export rand_addgaussiansnr
export rand_pitchshift
export rand_timestretch
export augment

"""
$(TYPEDSIGNATURES)

Random time sample shift of spectrograms `x`.

# Arguments
- x: time-series signals
- p: the probability of applying this transform
"""
function rand_timesampleshift(x::AbstractVector{T}; p = 0.5) where {T}
    if rand() ≤ p
        xn = size(x, 1)
        max_shift = xn ÷ 2
        shiftsample = rand(-max_shift:max_shift)
        circshift(x, (shiftsample,))
    else
        collect(x) # for type stability
    end
end

"""
$(TYPEDSIGNATURES)

Random flip `x` upside-down.

# Arguments
- x: time-serie signals
- p: the probability of applying this transform
"""
function rand_polarityinversion(x::AbstractVector{T}; p = 0.5) where {T}
    a = rand() ≤ p ? one(T) : -one(T)
    x .* a
end

"""
$(TYPEDSIGNATURES)

Random tanh distortion of `x`.

# Arguments
- x: time-series signals
- min_distortion: minimum "amount" of distortion to apply to the signal
- max_distortion: maximum "amount" of distortion to apply to the signal
- p: the probability of applying this transform

# Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/tanh_distortion.py
"""
function rand_tanhdistortion(x::AbstractVector{T}, 
                             min_distortion = T(0.01), 
                             max_distortion = T(0.7);
                             p = 0.5) where {T}
    if rand() ≤ p
        distortion_amount = (max_distortion - min_distortion) * rand(T) + min_distortion
        γ = 1 - 0.99 * distortion_amount
        threshold = quantile(abs.(x), γ) |> T
        gain_factor = T(0.5) / (threshold + T(1e-6))
        x_dist = tanh.(gain_factor .* x)
        x_rms = sqrt(mean(abs2, x))
        if x_rms > 1f-9
            x_dist_rms = sqrt(mean(abs2, x_dist))
            post_gain = x_rms / x_dist_rms
            x_dist .*= post_gain
        end
        x_dist
    else
        collect(x) # for type stability
    end
end

"""
Add Gaussian noise into `x`.

# Arguments
- x: time-series signal
- min_snr_db: minimum signal-to-noise ratio in dB
- max_snr_db: maximum signal-to-noise ratio in dB
- p: the probability of applying this transform

# Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/add_gaussian_snr.py
"""
function rand_addgaussiansnr(x::AbstractVector{T}, 
                             min_snr_db = T(0.0),
                             max_snr_db = T(30.0);
                             p = 0.5) where {T}
    if rand() ≤ p
        snr = rand(min_snr_db..max_snr_db)
        signal_rms = sqrt(mean(abs2, x))
        noise_rms = signal_rms / (10 ^ (snr / 20))
        x .+ noise_rms .* randn(T, length(x))
    else
        collect(x) # for type stability
    end
end

function rand_pitchshift end
function rand_timestretch end

"""
Apply augmentation `f` on each of the first dimension of `x`. 
"""
function augment(f::Function, x::AbstractArray) 
    stack(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        f(x1)
    end
end
