export reconstruct_image
function reconstruct_image(
        mv_m::Matrix{Tuple{Int64, Int64}},
        reference::AbstractArray{T,3},
        block_size::Int
    ) where T

    output = pad_image(zeros(T, size(reference)), block_size)

    Threads.@threads for idx in CartesianIndices(mv_m)
        i, j = idx[1], idx[2]
        dx, dy = mv_m[i, j]

        origin_y     = (i - 1) * block_size + 1
        origin_x     = (j - 1) * block_size + 1
        origin_y_end =  i      * block_size
        origin_x_end =  j      * block_size

        ref_y_start = origin_y + dy
        ref_x_start = origin_x + dx
        ref_y_end   = ref_y_start + block_size - 1
        ref_x_end   = ref_x_start + block_size - 1

        try
            output[:,    origin_y:origin_y_end, origin_x:origin_x_end] .= reference[:, ref_y_start:ref_y_end, ref_x_start:ref_x_end]
        catch e
            println("($i, $j) with vector [$dx, $dy]")
            println("error: ")
            display(e)

        end
    end
    return output
end
