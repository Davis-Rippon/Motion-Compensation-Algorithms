export exhaustive_search, logarithmic_search
using Base.Threads


function exhaustive_search(
        frame::AbstractMatrix{<:AbstractArray{T,3}},
        reference::AbstractArray{T,3},
        search_window::Int,
        block_size::Int = 16
    )::Matrix{Tuple{Int,Int}} where T

    reference = pad_image(reference, block_size)

    _, ref_y, ref_x = size(reference)
    ny, nx = size(frame)
    motion_vectors = Matrix{Tuple{Int,Int}}(undef, ny, nx)

    Threads.@threads for idx in CartesianIndices(frame)
        i, j = idx[1], idx[2]
        block = frame[i, j]

        origin_y = (i - 1) * block_size + 1
        origin_x = (j - 1) * block_size + 1

        best_cost = Inf
        best_dx, best_dy = 0, 0
        iter = Iterators.flatten((0:0, -search_window:-1, 1:search_window))

        for dx in iter
            for dy in iter

                ref_y_start = origin_y + dy
                ref_x_start = origin_x + dx
                ref_y_end   = ref_y_start + block_size - 1
                ref_x_end   = ref_x_start + block_size - 1

                if ref_y_start < 1 || ref_x_start < 1 ||
                    ref_y_end > ref_y || ref_x_end > ref_x
                    continue
                end

                candidate = view(reference, :, ref_y_start:ref_y_end, ref_x_start:ref_x_end)
                cost = mean_abs_difference(block, candidate)

                if cost == 0
                    best_cost = cost
                    best_dx, best_dy = dx, dy
                    @goto finish
                end

                if cost < best_cost
                    best_cost = cost
                    best_dx, best_dy = dx, dy
                end
            end
        end

        @label finish
        motion_vectors[i, j] = (best_dx, best_dy)
    end

    return motion_vectors
end


export logarithmic_search

function logarithmic_search(
        frame::AbstractMatrix{<:AbstractArray{T,3}},
        reference::AbstractArray{T,3},
        search_window::Int,
        block_size::Int = 16
    )::Matrix{Tuple{Int,Int}} where T
    reference = pad_image(reference, block_size)
    _, ref_y, ref_x = size(reference)
    ny, nx = size(frame)
    motion_vectors = Matrix{Tuple{Int,Int}}(undef, ny, nx)

    Threads.@threads for idx in CartesianIndices(frame)
        i, j = idx[1], idx[2]
        block = frame[i, j]
        origin_y = (i - 1) * block_size + 1
        origin_x = (j - 1) * block_size + 1

        center_dy, center_dx = 0, 0
        step = search_window ÷ 2

        best_cost = Inf
        best_dx, best_dy = 0, 0

        while step >= 1
            candidates = (
                (center_dy,        center_dx       ),
                (center_dy - step, center_dx       ),
                (center_dy + step, center_dx       ),
                (center_dy,        center_dx - step),
                (center_dy,        center_dx + step),
                (center_dy - step, center_dx - step),
                (center_dy - step, center_dx + step),
                (center_dy + step, center_dx - step),
                (center_dy + step, center_dx + step),
            )

            step_best_cost = Inf
            step_best_dx, step_best_dy = center_dx, center_dy

            for (dy, dx) in candidates
                ref_y_start = origin_y + dy
                ref_x_start = origin_x + dx
                ref_y_end   = ref_y_start + block_size - 1
                ref_x_end   = ref_x_start + block_size - 1

                if ref_y_start < 1 || ref_x_start < 1 ||
                   ref_y_end > ref_y || ref_x_end > ref_x
                    continue
                end

                candidate = view(reference, :, ref_y_start:ref_y_end, ref_x_start:ref_x_end)
                cost = mean_abs_difference(block, candidate)

                if cost == 0
                    best_dx, best_dy = dx, dy
                    @goto finish
                end

                if cost < step_best_cost
                    step_best_cost = cost
                    step_best_dx, step_best_dy = dx, dy
                end
            end

            if step_best_cost < best_cost
                best_cost = step_best_cost
                best_dx, best_dy = step_best_dx, step_best_dy
            end

            center_dx, center_dy = step_best_dx, step_best_dy
            step = step ÷ 2
        end

        @label finish
        motion_vectors[i, j] = (best_dx, best_dy)
    end

    return motion_vectors
end
