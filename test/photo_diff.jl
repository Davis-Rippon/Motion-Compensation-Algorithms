#!/usr/bin/env julia

using VideoIO
using FileIO
using Images

if length(ARGS) != 1
    println("Usage: julia extract_frames.jl <video_file>")
    exit(1)
end

video_file = ARGS[1]

reader = VideoIO.openvideo(video_file)
frame_count = counttotalframes(reader)
fps = 30 #VideoIO.framerate(reader)

length_seconds = Int64(frame_count) / fps

println("$(length_seconds) seconds")

seek(reader, length_seconds/2)
# Read first two frames
first = RGB{Float32}.(read(reader))
seek(reader, length_seconds / 8)
second = RGB{Float32}.(read(reader))

# Save the original frames
save("first_frame.png", first)
save("second_frame.png", second)

# Compute channel-wise difference (second - first)
# Clamp to [0,1] for PNG output.
difference = abs.(second .- first)

save("difference.png", difference)

println("Saved:")
println("  first_frame.png")
println("  second_frame.png")
println("  difference.png")
close(reader)
