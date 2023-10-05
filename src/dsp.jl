using SignalAnalysis 
using SignalAnalysis: SampledSignal

"""
$(SIGNATURES)

Lowpass-filter, downsample if `f2` is not `nothing`, followed by highpass-filter `s`.
"""
function filter_resample(s::SampledSignal{T}, f1, f2=nothing) where {T<:Real}
    s = s .- mean(s; dims = 1)
    if !isnothing(f2)
        lpf = fir(127, zero(f1), f2; fs = framerate(s))
        s = sfiltfilt(lpf, s)
        s = sresample(s, f2 / (framerate(s) / 2); dims = 1)
    end
    hpf = fir(127, f1; fs = framerate(s))
    T.(sfiltfilt(hpf, s))
end

function filter_resample(s::AbstractArray{T}, f1, f2=nothing; fs = 1) where {T<:Real}
    filter_resample(signal(s, fs), f1, f2) |> samples
end
