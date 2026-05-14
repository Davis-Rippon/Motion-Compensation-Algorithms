#%% Packages
using VideoIO, Images, Statistics, Plots
using StatsBase, ColorVectorSpace, Colors
include("src/block_match.jl")
using .BlockMatch

#%% Helper for padding 
function pad_to_block(img_array::AbstractArray{T,3}, block_size::Int) where T
    C, H, W = size(img_array)
    pad_h = (block_size - (H % block_size)) % block_size
    pad_w = (block_size - (W % block_size)) % block_size
    
    padded = zeros(T, C, H + pad_h, W + pad_w)
    padded[:, 1:H, 1:W] .= img_array
    return padded
end

#%% Robust Reconstruction
function reconstruct_image(
    mv_m::Matrix{Tuple{Int64, Int64}},
    reference::AbstractArray{T,3},
    block_size::Int
) where T
    C, H, W = size(reference)
    output = zeros(T, size(reference))

    Threads.@threads for idx in CartesianIndices(mv_m)
        i, j = idx[1], idx[2]
        dx, dy = mv_m[i, j] # Assumes search returns (dx, dy)

        # Destination coords
        out_y_s, out_x_s = (i-1)*block_size + 1, (j-1)*block_size + 1
        out_y_e, out_x_e = i*block_size, j*block_size

        # Clamp to image bounds
        out_y_e = min(out_y_e, H); out_x_e = min(out_x_e, W)
        bh, bw = out_y_e - out_y_s, out_x_e - out_x_s

        # Source reference coords with clamping to prevent "black holes"
        r_y_s = clamp(out_y_s + dy, 1, H - bh)
        r_x_s = clamp(out_x_s + dx, 1, W - bw)
        
        output[:, out_y_s:out_y_e, out_x_s:out_x_e] .= 
            reference[:, r_y_s:r_y_s+bh, r_x_s:r_x_s+bw]
    end
    return output
end

#%% Main Video Loop
BLOCK_SIZE = 16

function plot_graph(filename, search_method=0)
    video = VideoIO.openvideo(filename)
    prev_frame = read(video)
    
    residual_sums = Float64[]
    diff_sums     = Float64[]

    println("Processing frames to generate graph for $(basename(filename))...")

    # Process all frames without animating
    while !eof(video)
        frame_current = read(video)

        # 1. Pad both frames so they are identical sizes divisible by BLOCK_SIZE
        ref    = pad_to_block(float.(channelview(prev_frame)[1:3, :, :]), BLOCK_SIZE)
        target = pad_to_block(float.(channelview(frame_current)[1:3, :, :]), BLOCK_SIZE)

        # 2. Motion Search
        target_ba = construct_block_array(target, BLOCK_SIZE)
        if search_method == 0 
            motion_vector = exhaustive_search(target_ba, ref, 16, BLOCK_SIZE)
        else
            motion_vector = logarithmic_search(target_ba, ref, 16, BLOCK_SIZE)
        end

        # 3. Reconstruction
        reconstructed_raw = reconstruct_image(motion_vector, ref, BLOCK_SIZE)

        # 4. Math
        residual_mat = abs.(reconstructed_raw .- target)
        diff_mat     = abs.(target .- ref)

        push!(residual_sums, sum(residual_mat))
        push!(diff_sums,     sum(diff_mat))

        prev_frame = frame_current
    end
    close(video)

    # 5. Plotting the finalized graph
    method_name = search_method == 0 ? "Exhaustive Search" : "Logarithmic Search"
    
    p = plot(
        cumsum(residual_sums), 
        label="ME Residual (Cumulative)", 
        color=:blue, 
        linewidth=2,
        xlabel="Frame Number",
        ylabel="Cumulative Pixel Absolute Difference",
        title="Cumulative Error: $method_name\n$(basename(filename))",
        legend=:topleft,
        grid=true,
        size=(800, 500)
    )
    
    plot!(p, cumsum(diff_sums), label="Naive Frame Difference", color=:red, linewidth=2)

    # Save the plot
    prefix = search_method == 0 ? "exhaustive" : "logarithmic"
    save_filename = "$(prefix)_$(basename(filename))_graph.png"
    savefig(p, save_filename)
    
    # Calculate and output efficiency
    efficiency = 1 - (sum(residual_sums) / sum(diff_sums))
    println("Graph successfully saved as: $save_filename")
    println("Method $search_method Compression Efficiency: $(round(efficiency*100, digits=2))%")
    
    # Return the plot object in case you are running this in a REPL/Jupyter Notebook
    return p 
end

function plot_vals(filename, search_method=0)
    video = VideoIO.openvideo(filename)
    prev_frame = read(video)
    
    residual_sums = Float64[]
    diff_sums     = Float64[]

    # Use @animate to handle the frame buffer correctly
    anim = @animate while !eof(video)
        frame_current = read(video)

        # 1. Pad both frames so they are identical sizes divisible by BLOCK_SIZE
        ref    = pad_to_block(float.(channelview(prev_frame)[1:3, :, :]), BLOCK_SIZE)
        target = pad_to_block(float.(channelview(frame_current)[1:3, :, :]), BLOCK_SIZE)

        # 2. Motion Search
        target_ba = construct_block_array(target, BLOCK_SIZE)
        if search_method == 0 
            motion_vector = exhaustive_search(target_ba, ref, 16, BLOCK_SIZE)
        else
            motion_vector = logarithmic_search(target_ba, ref, 16, BLOCK_SIZE)
        end

        # 3. Reconstruction
        reconstructed_raw = reconstruct_image(motion_vector, ref, BLOCK_SIZE)

        # 4. Math (using padded versions for fair comparison)
        residual_mat = abs.(reconstructed_raw .- target)
        diff_mat     = abs.(target .- ref)

        push!(residual_sums, sum(residual_mat))
        push!(diff_sums,     sum(diff_mat))

        # 5. Visualization
        p1 = plot(colorview(RGB, residual_mat), title="Residual (Error)", axis=false)
        p2 = plot(colorview(RGB, target), title="Current Frame (Padded)", axis=false)
        p3 = plot(colorview(RGB, diff_mat), title="Naive Difference", axis=false)
        p4 = plot(cumsum(residual_sums), label="ME Residual", color=:blue, title="Cumulative Error")
        plot!(p4, cumsum(diff_sums), label="Naive Diff", color=:red)

        plot(p1, p3, p2, p4, layout=@layout([a b; c d]), size=(800, 600))

        prev_frame = frame_current
    end

    gif(anim, "$((search_method) == 0 ? "exhaustive" : "logarithmic")$(basename(filename))_.gif", fps=15)
    
    efficiency = 1 - (sum(residual_sums) / sum(diff_sums))
    println("Method $search_method Compression Efficiency: $(round(efficiency*100, digits=2))%")
    println("Sum of Residuals: $(sum(residual_sums))")
    println("Total Difference Sum: $(sum(diff_sums))")
    close(video)
end

@time plot_vals("data/videos/moving_square.mp4")

@time plot_graph("data/videos/growing_blue_circle.mp4", 1)
