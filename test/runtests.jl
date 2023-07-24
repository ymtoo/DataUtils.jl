using Test
using DataUtils

using CUDA
using DSP
using Flux
using FiniteDifferences
using MLUtils
using Statistics
#using TimeScaleModification

ondevices = [cpu, gpu]
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
            a = T.([1 2 3;
                    2 3 4;
                    3 4 5]) |> ondevice
            step = 1
            @test overlap_add(a, step) == T.([1,4,9,8,5]) |> ondevice
            step = 2
            @test overlap_add(a, step) == T.([1,2,5,3,7,4,5]) |> ondevice
            @inferred overlap_add(a, step)

            b = T.([1 2 3; 2 3 4; 3 4 5 ;;; 
                    1 2 3; 2 3 4; 3 4 5]) |> ondevice
            step = 1
            @test overlap_add(eachslice(b; dims = 3), step) == T.([1 2 3; 3 5 7; 5 7 9; 3 4 5]) |> ondevice 
            @test overlap_add(b, step) == T.(repeat([1,4,9,8,5]; outer = (1, 2))) |> ondevice # batched overlap add
            step = 2
            @test overlap_add(eachslice(b; dims = 3), step) == T.([1 2 3; 2 3 4; 4 6 8; 2 3 4; 3 4 5]) |> ondevice
            @test overlap_add(b, step) == T.(repeat([1,2,5,3,7,4,5]; outer = (1, 2))) |> ondevice # batched overlap add
            @inferred overlap_add(eachslice(b; dims = 3), step)
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
            rand_tanhdistortion,
            rand_addgaussiansnr,]
            # rand_pitchshift,
            # rand_timestretch]
    x = randn(Float32, 100, 2, batch_size)
    x1 = x[:,1,1]
    for aug ∈ augs
        @inferred aug(x1) 
        @test size(aug(x1)) == size(x1)
    end

    agg_augment(x) = x |> 
                     rand_timesampleshift |> 
                     rand_polarityinversion |>
                     rand_tanhdistortion |> 
                     rand_addgaussiansnr #|>
                    #  rand_pitchshift |>
                    #  rand_timestretch
    @inferred augment(agg_augment, x) 
    @test size(augment(agg_augment, x)) == size(x)

end 

@testset "losses" begin
    
    # Float32 throw an error
    # https://discourse.julialang.org/t/fftw-plan-applied-to-array-with-wrong-memory-alignment/101672
    T = Float64 
    N = 10000
    n = 128
    noverlap = 64
    windows = [nothing, hanning]
    for ondevice ∈ ondevices
        x = randn(T, N) 
        ondevice_x = ondevice(x)
        for window ∈ windows
            @test Array(DataUtils._stft(ondevice_x, n, noverlap; window = window)) ≈ 
                stft(x, n, noverlap; window = window)

            grad1 = gradient(x -> sum(abs2, DataUtils._stft(x, n, noverlap; window = window)), ondevice_x)[1] |> Array
            grad2 = grad(central_fdm(5, 1), 
                         x -> sum(abs2, stft(x, n, noverlap; window = window)), 
                         x)[1]
            @test cor(grad1, grad2) > 0.9

            stft_x = DataUtils._stft(ondevice_x, n, noverlap; window = window)

            epsilon = T(1e-10) + T(1e-10)im # avoid NaN if `stft_x̂` is zeros
            abs_stft_x = abs.(stft_x .+ epsilon)
            @test spectral_convergence_loss(abs_stft_x, abs_stft_x) == 0
            @test spectral_magnitude_loss(abs_stft_x, abs_stft_x, N) == 0
            @test stft_loss(x, x, n, noverlap, window) == 0
        end
    end

end
