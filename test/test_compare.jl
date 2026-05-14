#%% Getting BlockMatch
include("src/block_match.jl")

#%% Using
using Images, StatsBase, ColorVectorSpace, Colors, Plots
using .BlockMatch

#%%
black = channelview(load("data/black.png"))[1:3,:,:]
gray = channelview(load("data/gray.png"))[1:3,:,:]
white = channelview(load("data/white.png"))[1:3,:,:]
ref = channelview(load("data/hourglass.png"))[1:3,:,:]
target = channelview(load("data/hourglass-shifted.png"))[1:3,:,:]

c1 = channelview(load("data/circle1.png"))[1:3,:,:]
c2 = channelview(load("data/circle2.png"))[1:3,:,:]


complex_ba = construct_block_array(target)
black_ba = construct_block_array(black)

mv = exhaustive_search(complex_ba, ref, 16, 16)
bv = exhaustive_search(black_ba, black, 32, 16)

#%% Plot V field


function plot_vector_field(
    mv::Matrix{Tuple{T,T}},
    block_size::Real = 16
) where T

    rows, cols = size(mv)

    x_starts = Float64[]
    y_starts = Float64[]
    u_components = Float64[]
    v_components = Float64[]

    magnitudes = Float64[]

    for i in 1:rows
        for j in 1:cols

            dx, dy = mv[i, j]

            if dx != 0 || dy != 0

                x = (j - 0.5) * block_size
                y = (rows - i + 0.5) * block_size  # image coords (y down)

                push!(x_starts, x)
                push!(y_starts, y)

                push!(u_components, dx)
                push!(v_components, -dy)

                push!(magnitudes, sqrt(dx^2 + dy^2))
            end
        end
    end

    # Normalize magnitudes for visual scaling
    maxmag = maximum(magnitudes)

    linew = 0.5 .+ 3.0 .* (magnitudes ./ maxmag)
    colors = cgrad(:viridis)[magnitudes ./ maxmag]

    quiver(
        x_starts,
        y_starts,

        quiver = (u_components, v_components),

        xlims = (0, cols * block_size),
        ylims = (0, rows * block_size),

        aspect_ratio = :equal,

        xlabel = "x (pixels)",
        ylabel = "y (pixels)",
        title = "Motion Vector Field (Block Matching)",

        linewidth = linew,
        color = colors,

        size = (900, 700)
    )
end


#%% Reconstructing C2
target_ba = construct_block_array(c2)
cv = exhaustive_search(target_ba, c1, 100, 16)
plot_vector_field(cv)

x = reconstruct_image(cv, c1, 16)
img = colorview(RGB, x)
plot(img .- colorview(RGB, pad_image(c2,16)), title="Residual Of C2 Reconstruction ")

#%% Reconstructing c1
target_ba = construct_block_array(c1)
cv = exhaustive_search(target_ba, c2, 100, 16)
plot_vector_field(cv)

x = reconstruct_image(cv, c2, 16)
img = colorview(RGB, x)
plot(img .- colorview(RGB, pad_image(c1,16)), title="Residual of C1 Reconstruction")

#%% Reconstructing Complex_shifted
ref = channelview(load("data/hourglass.png"))[1:3,:,:]
target = channelview(load("data/hourglass-shifted.png"))[1:3,:,:]

target_ba = construct_block_array(target)
cv = logarithmic_search(target_ba, ref, 150, 16)
plot_vector_field(cv)

x = reconstruct_image(cv, ref, 16)
img = colorview(RGB, x)
plot(img)
plot(img .- colorview(RGB, pad_image(target,16)), title="Residual of C1 Reconstruction")

