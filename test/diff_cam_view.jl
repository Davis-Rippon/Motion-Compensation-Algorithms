#%% Packages
using GLMakie, VideoIO, Images, Statistics


#%% Stream Diff
camera = VideoIO.opencamera()
sleep(0.5)

im1 = read(camera)
diff0 = zeros(Float32, size(channelview(im1))[2:3])

obs = Observable(diff0)
fig, ax, hm = heatmap(obs, colormap=:grays)
display(fig)

is_running = true
@async while is_running
    im1 = read(camera)
    sleep(0.05)
    im2 = read(camera)

    diff = mean(Float32.(channelview(im1)) .- Float32.(channelview(im2)), dims=1)[1,:,:]
    obs[] = diff
end

#%% Shut down
close(camera)
is_running = false
