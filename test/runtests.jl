using Test
using DataUtils

using Flux
using MLUtils

ondevices = [cpu]
batch_size = 16

function _ismatch(x::AbstractMatrix, y::AbstractMatrix)
    n = size(x, 1)
    for i ∈ axes(y, 1)
        if x == y[i:i+n-1,:]
            return true
        end
    end
    false
end

function ismatch(x::AbstractMatrix, y::AbstractMatrix)
    if size(x, 1) < size(y, 1)
        _ismatch(x, y)
    else
        _ismatch(y, x)
    end
end

@testset "utils" begin
    
    Ts = [Int32, Int64, Float32, Float64]
    onvecvec = [false, true] 
    for ondevice ∈ ondevices
        for T ∈ Ts
            for onvecvec1 ∈ onvecvec 
                a = T.([1 2 3;
                        2 3 4;
                        3 4 5]) |> ondevice
                onvecvec1 && (a = eachcol(a))
                step = 1
                @test overlap_add(a, step) == T.([1,4,9,8,5]) |> ondevice
                step = 2
                @test overlap_add(a, step) == T.([1,2,5,3,7,4,5]) |> ondevice
            end
            b = T.([1 2 3; 2 3 4; 3 4 5 ;;; 
                    1 2 3; 2 3 4; 3 4 5]) |> ondevice
            step = 1
            @test overlap_add(b, step) == T.(repeat([1,4,9,8,5]; outer = (1, 2)))
            step = 2
            @test overlap_add(b, step) == T.(repeat([1,2,5,3,7,4,5]; outer = (1, 2)))
            @inferred overlap_add(b, step)
        end
    end

    xs = [randn(10, 3) for _ ∈ 1:batch_size]
    crop_xs = randomcrop_batch(xs, (5,))
    @test size(crop_xs) == (5, 3, batch_size)
    @test all([ismatch(crop_x, x) for (x, crop_x) 
               ∈ zip(xs, eachslice(crop_xs; dims = (3,)))])
    @test size(randomcrop_batch(xs, (5, 1))) == (5, 1, batch_size)
    @test size(randomcrop_batch(xs, (5, 2))) == (5, 2, batch_size)
    @test randomcrop_batch(xs, (10, 3)) == batch(xs)
    crop_xs = randomcrop_batch(xs, (11,))
    @test all([ismatch(crop_x, x) for (x, crop_x) 
               ∈ zip(xs, eachslice(crop_xs; dims = (3,)))])
    @test sum(iszero.(randomcrop_batch(xs, (11,)))) == batch_size * 3
    @test sum(iszero.(randomcrop_batch(xs, (11, 4)))) == batch_size * 14
    @inferred randomcrop_batch(xs, (11,))

end

@testset "augment" begin

    augs = [rand_timesampleshift, 
            rand_polarityinversion,
            rand_tanhdistortion]
    x = randn(Float32, 100, 2, batch_size)
    for aug ∈ augs
        @inferred aug(x) 
        @test size(aug(x)) == size(x)
    end

end 
