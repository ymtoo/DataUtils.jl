using JLD2: jldsave, load
using MLUtils
using Zygote

import Flux: cpu, reset!, state, loadmodel!

export overlap_add
export randomcrop_batch
export savemodel
export loadmodel!

"""
$(TYPEDSIGNATURES)

Reconstruct a sequence from time-shifted segments `x` (samples, frames). 
"""
function overlap_add(x::AbstractMatrix, step::Int)
    framelen, numframes = size(x)
    numsamples = step * (numframes - 1) + framelen
    y = Zygote.bufferfrom(zeros_like(x, (numsamples,))) # mutable when taking gradients
    for i ∈ 1:numframes
        startindex = (i - 1) * step + 1
        stopindex = startindex + framelen - 1
        @views y[startindex:stopindex] += x[:,i]
    end
    copy(y)
end

"""
$(TYPEDSIGNATURES)

Reconstruct sequences from time-shifted segments `x` (samples, frames, batch size). 
"""
function overlap_add(x::AbstractArray{T,3}, step::Int) where {T}
    framelen, numframes, batch_size = size(x)
    numsamples = step * (numframes - 1) + framelen
    y = Zygote.bufferfrom(zeros_like(x, (numsamples, batch_size))) # mutable when taking gradients
    for i ∈ 1:batch_size
        for j ∈ 1:numframes
            startindex = (j - 1) * step + 1
            stopindex = startindex + framelen - 1
            @views y[startindex:stopindex,i] += x[:,j,i]
        end
    end
    copy(y)
end

"""
$(TYPEDSIGNATURES)

Randomly crop `x`. The `cropsize` defines the length to crop for each dimension. 
The dimensions beyond `length(cropsize)` are skipped from the random cropping.
"""
function randomcrop_batch(x::AbstractArray{T}, cropsize) where {T}
    n = length(cropsize)
    dims = 1:n
    left_zeros_sizes = Int[]
    right_zeros_sizes = Int[]
    paddims = Int[]
    slices = UnitRange{Int}[]
    for dim ∈ dims
        dim_cropsize = cropsize[dim]
        if dim_cropsize > size(x, dim)
            startindex = 1
            stopindex = size(x, dim)
            left_zeros_size = rand(1:(dim_cropsize - size(x, dim)))
            right_zeros_size = (dim_cropsize - size(x, dim)) - left_zeros_size
            push!(left_zeros_sizes, left_zeros_size)
            push!(right_zeros_sizes, right_zeros_size)
            push!(paddims, dim)
        else
            startindex = rand(1:(size(x, dim) - dim_cropsize + 1))
            stopindex = startindex + dim_cropsize - 1
            push!(left_zeros_sizes, size(x, dim))
            push!(right_zeros_sizes, size(x, dim))
        end
        push!(slices, startindex:stopindex)
    end
    for remain_dim ∈ n+1:ndims(x)
        push!(slices, 1:size(x, remain_dim))
        push!(left_zeros_sizes, size(x, remain_dim))
        push!(right_zeros_sizes, size(x, remain_dim))
    end
    crop_x = getindex(x, slices...)::Array{T}
    if !isempty(paddims) 
        left_z = zeros_like(x, Tuple(left_zeros_sizes))::Array{T}
        right_z = zeros_like(x, Tuple(right_zeros_sizes))::Array{T}
        cat(left_z, crop_x, right_z; dims = paddims)::Array{T}
    else
        crop_x
    end
end

"""
$(TYPEDSIGNATURES)

Randomly crop each element of `xs`. The `cropsize` defines the length to crop for each dimension. 
The dimensions beyond `length(cropsize)` are skipped from the random cropping.
"""
function randomcrop_batch(xs::AbstractVector{T}, cropsize) where {T}
    map(xs) do x
        randomcrop_batch(x, cropsize)::T
    end |> batch
end

"""
$(SIGNATURES)

Save Flux model. 
"""
function savemodel(savepath::AbstractString, model)
    model_state = reset!(cpu(model)) |> state
    jldsave(savepath; model_state)
end

"""
$(SIGNATURES)

Load Flux model.
"""
function loadmodel!(init_model, loadpath::AbstractString)
    model_state = load(loadpath, "model_state")
    loadmodel!(init_model, model_state)
    init_model
end
