using Test
using DataUtils

using Flux
using MLUtils

ondevices = [cpu]
batch_size = 16

@testset "utils" begin

    Ts = [Int32, Int64, Float32, Float64]

    for ondevice ∈ ondevices
        for T ∈ Ts
            a = T.([1 2 3;
                    2 3 4;
                    3 4 5]) |> ondevice
            step = 1
            @test overlap_add(a, step) == T.([1,4,9,8,5]) |> ondevice
            step = 2
            @test overlap_add(a, step) == T.([1,2,5,3,7,4,5]) |> ondevice

            b = T.([1 2 3; 2 3 4; 3 4 5 ;;; 
                    1 2 3; 2 3 4; 3 4 5])
            step = 1
            @test overlap_add(b, step) == T.(repeat([1,4,9,8,5]; outer = (1, 2)))
            step = 2
            @test overlap_add(b, step) == T.(repeat([1,2,5,3,7,4,5]; outer = (1, 2)))
        end
    end

    xs = [randn(10, 3) for _ ∈ 1:batch_size]
    @test size(randomcrop_batch(xs, (5,))) == (5, 3, batch_size)
    @test size(randomcrop_batch(xs, (5, 1))) == (5, 1, batch_size)
    @test size(randomcrop_batch(xs, (5, 2))) == (5, 2, batch_size)
    @test randomcrop_batch(xs, (10, 3)) == batch(xs)
    crop_xs = randomcrop_batch(xs, (11,))
    @test sum(iszero.(randomcrop_batch(xs, (11,)))) == batch_size * 3
    @test sum(iszero.(randomcrop_batch(xs, (11, 4)))) == batch_size * 14

end
