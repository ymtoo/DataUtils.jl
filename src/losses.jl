using ChainRulesCore: @ignore_derivatives
using DSP
using DSP.Periodograms: compute_window, forward_plan

export stft_loss
export spectral_convergence_loss
export spectral_magnitude_loss

function _stft(x::AbstractVector{T}, 
               n::Int, noverlap::Int; 
               window::Union{Nothing,W}=nothing) where {T,W}
    Tisreal = T <: Real
    numsamples = length(x)
    numframes = @ignore_derivatives numsamples >= n ? div((numsamples - n), n - noverlap) + 1 : 0
    inputtype = @ignore_derivatives fftintype(T)
    outtype = @ignore_derivatives fftouttype(T)

    win, norm2 = @ignore_derivatives compute_window(window, n)
    !isnothing(win) && (win = @ignore_derivatives inputtype.(typeof(x)(win)))

    oneton = @ignore_derivatives Base.OneTo(n)
    tmp = @ignore_derivatives similar(x, outtype, Tisreal ? (n >> 1)+1 : n)
    plan = @ignore_derivatives forward_plan(zeros_like(x, inputtype, n), tmp)
    map(1:numframes) do i
        offset = @ignore_derivatives (i - 1) * (n - noverlap)
        x1 = inputtype.(isnothing(win) ? x[offset .+ oneton] : x[offset .+ oneton] .* win)
        plan * x1
    end |> stack
end

"""
STFT loss.
"""
function stft_loss(x̂::AbstractVector, x::AbstractVector, nfft, noverlap, window = nothing)
    numsamples = length(x̂)
    stft_x̂ = _stft(x̂, nfft, noverlap; window = window)
    stft_x = _stft(x, nfft, noverlap; window = window)
    epsilon = @ignore_derivatives T(1e-10) + T(1e-10)im # avoid NaN if `stft_x̂` is zeros
    abs_stft_x̂ = abs.(stft_x̂ .+ epsilon)
    abs_stft_x = abs.(stft_x .+ epsilon)
    spectral_convergence_loss(abs_stft_x̂, abs_stft_x) + spectral_magnitude_loss(abs_stft_x̂, abs_stft_x, numsamples)
end

function stft_loss(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, nfft, noverlap, window = nothing) where {T}
    _, ax2, ax3 = @ignore_derivatives axes(x)
    ijs = @ignore_derivatives reshape([(i,j) for i ∈ ax2 for j ∈ ax3], length(ax2), length(ax3))
    map(ijs) do ij
        i, j = ij
        x̂1 = x̂[:,i,j]
        x1 = x[:,i,j]
        stft_loss(x̂1, x1, nfft, noverlap, window)
    end 
end

# https://github.com/JuliaDiff/ChainRules.jl/issues/722
# function stft_loss(x̂::AbstractArray, x::AbstractArray, nfft, noverlap)
#     remain_dims = tuple(2:ndims(x)...)
#     map(zip(eachslice(x; dims = remain_dims), eachslice(x̂; dims = remain_dims))) do (x1, x̂1)
#         stft_loss(x1, x̂1, nfft, noverlap)
#     end |> stack
# end

function spectral_convergence_loss(abs_stft_x̂::AbstractMatrix{T}, 
                                   abs_stft_x::AbstractMatrix{T}) where {T}
    sqrt.(sum(abs2, abs_stft_x̂ - abs_stft_x)) ./ sqrt(sum(abs2, abs_stft_x))
end

function spectral_magnitude_loss(abs_stft_x̂::AbstractMatrix{T}, 
                                 abs_stft_x::AbstractMatrix{T}, 
                                 numsamples::Int) where {T}
    sum(abs, log.(abs_stft_x̂) - log.(abs_stft_x)) / numsamples
end
