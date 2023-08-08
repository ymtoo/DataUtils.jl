module DataUtilsTimeScaleModificationExt

using DataUtils
using IntervalSets
using TimeScaleModification: pitchshift, timestretch, WSOLA, hanning

"""
Random pitch shifting of `x`.

# Arguments
- x: time-series signal
- max_semitones: maximum semitones to shift
- n: window size
- synhopsize: hop size of the synthesis window
- p: the probability of applying this transform
"""
function DataUtils.rand_pitchshift(x::AbstractVector{T}, max_semitones::Real = 12, n::Int = 128, synhopsize::Int = 64; p = 0.5) where {T}
    if rand() ≤ p
        semitones = rand(-max_semitones..max_semitones)
        pitchshift(WSOLA(n, synhopsize, hanning, 10), x, semitones)
    else
        collect(x) # for type stability
    end
end

"""
Random time stretching of `x`.

# Arguments
- x: time-series signal
- max_shiftedspeed: maximum shited speed
- n: window size
- synhopsize: hop size of the synthesis window
- p: the probability of applying this transform
"""
function DataUtils.rand_timestretch(x::AbstractVector{T}, max_shiftedspeed::Real = 0.1, n::Int = 128, synhopsize::Int = 64; p = 0.5) where {T}
    if rand() ≤ p
        speed = rand((1-max_shiftedspeed)..(1+max_shiftedspeed))
        timestretch(WSOLA(n, synhopsize, hanning, 10), x, speed)
    else
        collect(x) # for type stability
    end
end

end
