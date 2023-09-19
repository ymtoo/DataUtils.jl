using JLD2: jldsave, load
using MLUtils
using Zygote

import Flux: cpu, reset!, state, loadmodel!

export overlap_add
export randomcrop_batch
export savemodel, savemodelopt
export loadmodel!, loadopt

"""
$(TYPEDSIGNATURES)

Reconstruct a sequence from time-shifted segments `x` vector of vector where the length of the 
outer vector is number of segments and the length of the inner vector is number of samples 
of the segment. 
"""
function overlap_add(x::AbstractVector{T}, step::Int) where {T}
    numframes = length(x)
    segment_sizes = size.(x)
    remain_segment_dims = first(segment_sizes)[2:end]
    remain_dims = (1:remain_segment_dim for remain_segment_dim ∈ remain_segment_dims) 
    numsamples = step * (numframes - 1) + first(last(segment_sizes))
    y = Zygote.bufferfrom(zeros_like(first(x), (numsamples,remain_segment_dims...))) # mutable when taking gradients
    for (i, x1) ∈ enumerate(x)
        startindex = (i - 1) * step + 1
        stopindex = startindex + first(segment_sizes[i]) - 1
        slice_dims = (startindex:stopindex, remain_dims...)
        y[slice_dims...] += x1
    end
    copy(y)
end

"""
$(TYPEDSIGNATURES)

Reconstruct a sequence from time-shifted segments `x` matrix where the number of columns is
number of segments and the number of rows is number of samples of each segment. 
"""
overlap_add(x::AbstractMatrix{T}, step::Int) where {T<:Real}= overlap_add(eachcol(x), step)

"""
$(TYPEDSIGNATURES)

Reconstruct a sequence from batched time-shifted segments `x` 3D array where the first dimension is
number of samples of a segment, the second dimension is number of segments and the third 
dimension is batch size. 
"""
function overlap_add(x::AbstractArray{T,3}, step::Int) where {T<:Real} 
    stack(eachslice(x; dims =3)) do x1
        overlap_add(x1, step)
    end
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
function randomcrop_batch(xs::AbstractVector{T}, cropsize) where {T<:AbstractArray}
    stack(xs) do x
        randomcrop_batch(x, cropsize)::T
    end
end

"""
$(SIGNATURES)

Save Flux model. 
"""
function savemodel(savepath::AbstractString, model)
    reset!(model)
    model_state = cpu(model) |> state
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

"""
$(SIGNATURES)

Save Flux model and optimiser state. 
"""
function savemodelopt(savepath::AbstractString, model, opt_state)
    reset!(model)
    model_state = cpu(model) |> state
    opt_state = cpu(opt_state)
    jldsave(savepath; model_state, opt_state)
end

"""
$(SIGNATURES)

Load Flux optimiser state.
"""
function loadopt(loadpath::AbstractString)
    load(loadpath, "opt_state")
end
