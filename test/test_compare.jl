#%% Getting BlockMatch
include("../src/block_match.jl")

#%%
using .BlockMatch

b1::Matrix{Float32} = [1 2; 3 4]
b2::Matrix{Float32} = [1 2; 3 4]

comp(b1, b2)
