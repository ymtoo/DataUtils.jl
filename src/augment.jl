using Statistics

export rand_timesampleshift
export rand_polarityinversion
export rand_tanhdistortion

"""
$(TYPEDSIGNATURES)

Random time sample shift of spectrograms `x`.

# Arguments
- x: multi-channel signals
- p: the probability of applying this transform
"""
function rand_timesampleshift(x::AbstractArray; p = 0.5)
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        if rand() > p
            xn = size(x1, 1)
            max_shift = xn ÷ 2
            shiftsample = rand(-max_shift:max_shift)
            circshift(x1, (shiftsample,))::typeof(x1)
        else
            x1
        end
    end |> stack
end

"""
$(TYPEDSIGNATURES)

Random flip `x` upside-down.

# Arguments
- x: multi-channel signals
- p: the probability of applying this transform
"""
function rand_polarityinversion(x::AbstractArray{T}; p = 0.5) where {T}
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        a = rand() ≥ p ? one(T) : -one(T)
        x1 .* a
    end |> stack
end

"""
$(TYPEDSIGNATURES)

Random tanh distortion of `x`.

# Arguments
- x: multi-channel signals
- min_distortion: minimum "amount" of distortion to apply to the signal
- max_distortion: maximum "amount" of distortion to apply to the signal
- p: the probability of applying this transform

Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/tanh_distortion.py
"""
function rand_tanhdistortion(x::AbstractArray{T}, 
                             min_distortion = T(0.01), 
                             max_distortion = T(0.7);
                             p = 0.5) where {T}
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        if rand() > p
            distortion_amount = (max_distortion - min_distortion) * rand(T) + min_distortion
            γ = 1 - 0.99 * distortion_amount
            threshold = quantile(abs.(x1), γ) |> T
            gain_factor = T(0.5) / (threshold + T(1e-6))
            dist_x1 = tanh.(gain_factor .* x1)
            rms_x1 = sqrt(mean(abs2, x1))
            if rms_x1 > 1f-9
                rms_dist_x1 = sqrt(mean(abs2, dist_x1))
                post_gain = rms_x1 / rms_dist_x1
                dist_x1 .*= post_gain
            end
            dist_x1::typeof(x1)
        else
            x1
        end
    end |> stack
end
