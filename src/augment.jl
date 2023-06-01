using Statistics

export rand_timesampleshift
export rand_polarityinversion
export rand_tanhdistortion

"""
$(TYPEDSIGNATURES)

Random time sample shift of spectrograms `x`.
"""
function rand_timesampleshift(x::AbstractArray)
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        xn = size(x1, 1)
        max_shift = xn ÷ 2
        shiftsample = rand(-max_shift:max_shift)
        circshift(x1, (shiftsample,))
    end |> stack
end

"""
$(TYPEDSIGNATURES)

Random flip `x` upside-down.
"""
function rand_polarityinversion(x::AbstractArray{T}) where {T}
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        a = rand() ≥ 0.5 ? one(T) : -one(T)
        x1 .* a
    end |> stack
end

"""
$(TYPEDSIGNATURES)

Random tanh distortion of `x`.

Reference
https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/tanh_distortion.py
"""
function rand_tanhdistortion(x::AbstractArray{T}, min_distortion = T(0.01), max_distortion = T(0.7)) where {T}
    map(eachslice(x; dims = tuple(2:ndims(x)...))) do x1
        @show size(x1)
        if rand() > 0.5
            distortion_amount = (max_distortion - min_distortion) * rand(T) + min_distortion
            p = 1 - 0.99 * distortion_amount
            threshold = quantile(abs.(x1), p) |> T
            gain_factor = T(0.5) / (threshold + T(1e-6))
            dist_x1 = tanh.(gain_factor .* x1)
            rms_x1 = sqrt(mean(abs2, x1))
            if rms_x1 > 1f-9
                rms_dist_x1 = sqrt(mean(abs2, dist_x1))
                post_gain = rms_x1 / rms_dist_x1
                dist_x1 .*= post_gain
            end
            dist_x1
        else
            x1
        end
    end |> stack
end
