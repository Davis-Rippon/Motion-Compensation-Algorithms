#%% Getting BlockMatch
include("src/block_match.jl")

#%% Using
using Images, StatsBase, ColorVectorSpace, Colors
using .BlockMatch

#%%
black = channelview(load("data/black.png"))[1:3,:,:]
gray = channelview(load("data/gray.png"))[1:3,:,:]
white = channelview(load("data/white.png"))[1:3,:,:]
complex_original = channelview(load("data/hourglass.png"))[1:3,:,:]
complex_shifted = channelview(load("data/hourglass-shifted.png"))[1:3,:,:]

complex_ba = construct_block_array(complex_shifted)
black_ba = construct_block_array(black)

mv = exhaustive_search(complex_ba, complex_original, 16, 16)
exhaustive_search(black_ba, black, 32, 16)

#%% Plot V field
using Plots


function plot_vector_field(mv::Matrix{Tuple{Float64, Float64}})
    rows, cols = size(mv)
    
    # Collect starting points and vector components
    x_starts = Float64[]
    y_starts = Float64[]
    u_components = Float64[]
    v_components = Float64[]
    
    for i in 1:rows
        for j in 1:cols
            dx, dy = mv[i, j]
            if dx != 0 || dy != 0  # Only plot non-zero vectors
                push!(x_starts, j)   # j = column = x-axis
                push!(y_starts, i)   # i = row = y-axis
                push!(u_components, dx)
                push!(v_components, dy)
            end
        end
    end
    
    quiver(x_starts, y_starts,
        quiver = (u_components, v_components),
        xlims = (0, cols + 1),
        ylims = (0, rows + 1),
        xlabel = "Column (j)",
        ylabel = "Row (i)",
        title = "Vector Field",
        arrow = true,
        linewidth = 0.5,
        lengthscale =0.1,
        arrowscale = 0.1,
        color = :blue,
        aspect_ratio = :equal,
        size = (800, 600)
    )
end

mv = mv.*0.1
plot_vector_field(mv)
