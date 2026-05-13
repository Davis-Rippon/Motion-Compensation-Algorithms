#%% Packages
using VideoIO
using Images
include("src/block_match.jl")
using VideoIO, Images, Statistics, Plots
using Images, StatsBase, ColorVectorSpace, Colors
using .BlockMatch

#%% Main
BLOCK_SIZE = 16

function plot_vals(filename, search_method=0)
video = VideoIO.openvideo(filename)
    prev_frame = read(video)
    residual_sums = Float64[]
    diff_sums     = Float64[]

    @gif while !eof(video)
        frame_current = read(video)

        ref    = float.(channelview(prev_frame)[1:3, :, :])
        target = float.(channelview(frame_current)[1:3, :, :])

        target_ba     = construct_block_array(target, BLOCK_SIZE)
        if search_method == 0 
            motion_vector = exhaustive_search(target_ba, ref, 16, BLOCK_SIZE)
        else
            motion_vector = logarithmic_search(target_ba, ref, 16, BLOCK_SIZE)
        end
        x             = reconstruct_image(motion_vector, ref, BLOCK_SIZE)

        reconstructed = colorview(RGB, x)
        padded_target = colorview(RGB, pad_image(target, BLOCK_SIZE))

        residual  = colorview(RGB, abs.(channelview(reconstructed) .- channelview(padded_target)))
        full_diff = colorview(RGB, clamp.(abs.(float.(channelview(frame_current)) .- float.(channelview(prev_frame))), 0f0, 1f0))

        push!(residual_sums, sum(channelview(residual)))
        push!(diff_sums,     sum(channelview(full_diff)))

        p1 = plot(residual,      title="Residual of Target Reconstruction", titlefontsize=7,
                  guidefontsize=6, tickfontsize=5, legendfontsize=6,
                  axis=false, ticks=false, aspect_ratio=:equal)
        p2 = plot(frame_current, title="Reference Video", titlefontsize=7,
                  guidefontsize=6, tickfontsize=5, legendfontsize=6,
                  axis=false, ticks=false, aspect_ratio=:equal)
        p3 = plot(full_diff,     title="Full Difference", titlefontsize=7,
                  guidefontsize=6, tickfontsize=5, legendfontsize=6,
                  axis=false, ticks=false, aspect_ratio=:equal)
        p4 = plot(cumsum(residual_sums), label="Residual", color=:blue,
                  title="Cumulative Pixel Sum", titlefontsize=7,
                  xlabel="Frame", ylabel="Cumulative Sum",
                  guidefontsize=6, tickfontsize=5, legendfontsize=6,
                  legend=:topleft)
        plot!(p4, cumsum(diff_sums), label="Full Diff", color=:red)

        plot(p1, p3, p2, p4, layout=@layout([a b; c d]))

        prev_frame = frame_current
    end

    f_rs = sum(residual_sums)
    f_ds = sum(diff_sums)

    efficiency = 1 - f_rs/f_ds

    println("Final Residual: $f_rs")
    println("Final Diff: $f_ds")
    println("Compression: $efficiency")
    println("Search Method: $search_method")
    close(video)
end

@time plot_vals("data/videos/moving_square.mp4")
@time plot_vals("data/videos/moving_square.mp4", 1)

# plot_vals("data/videos/globe_shortened.mp4")
# plot_vals("data/videos/globe.mp4") # I just did this one second so this will be what is 
# plot_vals("data/videos/globe.mp4", 1)
