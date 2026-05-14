export reconstruct_image

function reconstruct_image(
        mv_m::Matrix{Tuple{Int64, Int64}},
        reference::AbstractArray{T,3},
        block_size::Int
    ) where T

    C, H, W = size(reference)
    
    # Initialize output to exactly match the reference size
    output = zeros(T, size(reference))

    Threads.@threads for idx in CartesianIndices(mv_m)
        i, j = idx[1], idx[2]
        dx, dy = mv_m[i, j]

        # 1. Output destination coordinates
        out_y_start = (i - 1) * block_size + 1
        out_x_start = (j - 1) * block_size + 1
        out_y_end   = i * block_size
        out_x_end   = j * block_size

        # Protect against non-divisible image sizes at the bottom/right edges
        out_y_end = min(out_y_end, H)
        out_x_end = min(out_x_end, W)

        # Actual height and width of this specific block (usually `block_size`, 
        # but smaller at the edges if the image isn't perfectly divisible)
        block_h = out_y_end - out_y_start
        block_w = out_x_end - out_x_start

        # 2. Source reference coordinates (where to pull from)
        ref_y_start = out_y_start + dy
        ref_x_start = out_x_start + dx

        # 3. CLAMPING (Replaces your try...catch)
        # This prevents the BoundsError and ensures edge blocks pull from the closest valid edge 
        # instead of failing and leaving a black square.
        r_y_s = clamp(ref_y_start, 1, H - block_h)
        r_x_s = clamp(ref_x_start, 1, W - block_w)
        
        r_y_e = r_y_s + block_h
        r_x_e = r_x_s + block_w

        # 4. Write to output
        output[:, out_y_start:out_y_end, out_x_start:out_x_end] .= 
            reference[:, r_y_s:r_y_e, r_x_s:r_x_e]
    end
    
    return output
end
