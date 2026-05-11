export mean_abs_difference, padded, construct_block_array, pad_image

using ImageFiltering, StatsBase

"""
Construct matrix of blocks

Expects a frame with 3 colour channels 
"""
function construct_block_array(frame::AbstractArray{T, 3}, block_size::Int = 16)::AbstractMatrix{<:AbstractArray{T,3}} where T
    fsize::Tuple{Int, Int, Int} = size(frame)
    _, x, y = fsize

    pad_x = mod(block_size - (x % block_size), block_size)
    pad_y = mod(block_size - (y % block_size), block_size)

    frame_padded = padded(pad_x, pad_y, frame)

    _, padded_y, padded_x = size(frame_padded)

    ny::Int = padded_y / block_size
    nx::Int = padded_x / block_size
    blocks = [
              view(
                   frame_padded, 
                   :,                                          
                   (i - 1) * block_size + 1 : i * block_size,  
                   (j - 1) * block_size + 1 : j * block_size   
                  )
              for i in 1:ny, j in 1:nx
             ]

    return blocks
end

"""
Compare two blocks
"""
function mean_abs_difference(b1::AbstractArray{T}, b2::AbstractArray{T}):: Float64 where T
    return mean(abs.(Float64.(b1) .- Float64.(b2)))
end

function pad_image(frame::AbstractArray{T, 3}, block_size::Int = 16)::AbstractArray{T, 3} where T
    fsize::Tuple{Int, Int, Int} = size(frame)
    _, x, y = fsize

    pad_x = mod(block_size - (x % block_size), block_size)
    pad_y = mod(block_size - (y % block_size), block_size)

    return padded(pad_x, pad_y, frame)
end

"""
Pad block
"""
function padded(pad_x::Int, pad_y::Int, block::AbstractArray{T, 3})::AbstractArray{T, 3} where T
    return padarray(block, Pad(:replicate, (0,0,0), (0, pad_x, pad_y)))
end
