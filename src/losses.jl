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
    x1 = zeros_like(x, inputtype, n)

    win, norm2 = @ignore_derivatives compute_window(window, n)
    !isnothing(win) && (win = @ignore_derivatives inputtype.(typeof(x)(win)))

    oneton = Base.OneTo(n)
    tmp = similar(x, outtype, Tisreal ? (n >> 1)+1 : n)
    plan = @ignore_derivatives forward_plan(x1, tmp)
    map(1:numframes) do i
        offset = (i - 1) * (n - noverlap)
        @views x1 = isnothing(window) ? x[offset .+ oneton] : x[offset .+ oneton] .* win
        plan * x1
    end |> stack
end

"""
STFT loss.
"""
function stft_loss(x̂::AbstractVector, x::AbstractVector, nfft, noverlap)
    numsamples = length(x̂)
    stft_x̂ = _stft(x̂, nfft, noverlap)
    stft_x = _stft(x, nfft, noverlap)
    spectral_convergence_loss(stft_x̂, stft_x) + spectral_magnitude_loss(stft_x̂, stft_x, numsamples)
end

function stft_loss(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, nfft, noverlap) where {T}
    _, ax2, ax3 = axes(x)
    map(ax3) do i 
        map(ax2) do j
            @views x̂1 = x̂[:,j,i]
            @views x1 = x[:,j,i]
            stft_loss(x1, x̂1, nfft, noverlap)
        end |> stack
    end |> stack
end

# https://github.com/JuliaDiff/ChainRules.jl/issues/722
# function stft_loss(x̂::AbstractArray, x::AbstractArray, nfft, noverlap)
#     remain_dims = tuple(2:ndims(x)...)
#     map(zip(eachslice(x; dims = remain_dims), eachslice(x̂; dims = remain_dims))) do (x1, x̂1)
#         stft_loss(x1, x̂1, nfft, noverlap)
#     end |> stack
# end

function spectral_convergence_loss(stft_x̂::AbstractMatrix, stft_x::AbstractMatrix)
    abs_stft_x = abs.(stft_x)
    sqrt.(sum(abs2, abs.(stft_x̂) - abs_stft_x)) ./ sqrt.(sum(abs2, abs_stft_x))
end

function spectral_magnitude_loss(stft_x̂::AbstractMatrix, stft_x::AbstractMatrix, numsamples::Int)
    sum(abs, log.(abs.(stft_x̂)) - log.(abs.(stft_x))) / numsamples
end
