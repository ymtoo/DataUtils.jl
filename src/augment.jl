using IntervalSets
using Statistics

export rand_timesampleshift
export rand_polarityinversion
export rand_tanhdistortion
export rand_addgaussiansnr

"""
$(TYPEDSIGNATURES)

Random time sample shift of spectrograms `x`.

# Arguments
- x: multi-channel signals
- p: the probability of applying this transform
"""
function rand_timesampleshift(x::AbstractArray{T}; p = 0.5) where {T}
    stack(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        if rand() ≤ p
            xn = size(x1, 1)
            max_shift = xn ÷ 2
            shiftsample = rand(-max_shift:max_shift)
            circshift(x1, (shiftsample,))
        else
            collect(x1) # for type stability
        end
    end
end

"""
$(TYPEDSIGNATURES)

Random flip `x` upside-down.

# Arguments
- x: multi-channel signals
- p: the probability of applying this transform
"""
function rand_polarityinversion(x::AbstractArray{T}; p = 0.5) where {T}
    stack(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        a = rand() ≤ p ? one(T) : -one(T)
        x1 .* a
    end
end

"""
$(TYPEDSIGNATURES)

Random tanh distortion of `x`.

# Arguments
- x: multi-channel signals
- min_distortion: minimum "amount" of distortion to apply to the signal
- max_distortion: maximum "amount" of distortion to apply to the signal
- p: the probability of applying this transform

# Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/tanh_distortion.py
"""
function rand_tanhdistortion(x::AbstractArray{T}, 
                             min_distortion = T(0.01), 
                             max_distortion = T(0.7);
                             p = 0.5) where {T}
    stack(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        if rand() ≤ p
            distortion_amount = (max_distortion - min_distortion) * rand(T) + min_distortion
            γ = 1 - 0.99 * distortion_amount
            threshold = quantile(abs.(x1), γ) |> T
            gain_factor = T(0.5) / (threshold + T(1e-6))
            x1_dist = tanh.(gain_factor .* x1)
            x1_rms = sqrt(mean(abs2, x1))
            if x1_rms > 1f-9
                x1_dist_rms = sqrt(mean(abs2, x1_dist))
                post_gain = x1_rms / x1_dist_rms
                x1_dist .*= post_gain
            end
            x1_dist
        else
            collect(x1) # for type stability
        end
    end
end

"""
Add Gaussian noise into `x`.

# Arguments
- x: multi-channel signals
- min_snr_db: minimum signal-to-noise ratio in dB
- max_snr_db: maximum signal-to-noise ratio in dB
- p: the probability of applying this transform

# Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/add_gaussian_snr.py
"""
function rand_addgaussiansnr(x::AbstractArray{T}, 
                             min_snr_db = T(0.0),
                             max_snr_db = T(30.0);
                             p = 0.5) where {T}
    stack(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        if rand() ≤ p
            snr = rand(min_snr_db..max_snr_db)
            signal_rms = sqrt(mean(abs2, x1))
            noise_rms = signal_rms / (10 ^ (snr / 20))
            x1 .+ noise_rms .* randn(T, length(x1))
        else
            collect(x1) # for type stability
        end
    end
end
