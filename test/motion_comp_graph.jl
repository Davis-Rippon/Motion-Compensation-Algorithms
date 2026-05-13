#%% Packages
include("src/block_match.jl")
using GLMakie, VideoIO, Images, Statistics
using Images, StatsBase, ColorVectorSpace, Colors
using .BlockMatch

camera = VideoIO.opencamera()
sleep(0.5)
im1 = read(camera)
diff0 = zeros(Float32, size(channelview(im1))[2:3])
obs = Observable(diff0)
fig, ax, hm = heatmap(obs, colormap=:grays)
display(fig)

is_running = true
errormonitor(@async while is_running
    println("1. Read Camera")
    im1 = read(camera)
    sleep(0.05)
    println("2. Read Camera")
    im2 = read(camera)
    println("3. Create Channel Views")
    ref    = float.(channelview(im1))
    target = float.(channelview(im2))
    println("4. Construct BA")
    target_block_array = construct_block_array(target)
    println("5. Ex Search")
    mv = exhaustive_search(target_block_array, ref, 32, 16)
    println("6. Reconstruct")
    im_channelview = reconstruct_image(mv, ref, 16)
    img = colorview(RGB, im_channelview)
    println("7. Calculate Difference")
    diff_reconstruction = img .- colorview(RGB, pad_image(target, 16))
    diff = colorview(RGB, float.(channelview(im1)) .- float.(channelview(im2)))
    println("8. Assign")
    intensity = mean(abs.(channelview(diff) .- channelview(diff_reconstruction)), dims=1)[1,:,:]
    obs[] = Float32.(intensity)
end)

#%% Shut down
close(camera)
is_running = false
