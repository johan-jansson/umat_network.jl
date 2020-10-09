# Recurrent neural network for 1D mixed kinematic hardening plasticity subject to cyclic loading
# Chain(LSTM(timesteps, timesteps*2),Dense(timesteps*2, timesteps)) with loss 1e-5 after 5 epochs
# 9000 models, 90 timesteps, E = 70000,  H = 20000,  σʸ = 100, r = 0.5
using Flux, BSON


BSON.@load "network.bson" network

function nn_normalize_two(x) # x ∈ ℜ -> x ∈ [-1,...,1]
    x_min = findmin(x)[1]
    x_max = findmax(x)[1]
    return ((x .- ((x_min+x_max)/2))/((x_max-x_min)/2)), x_min, x_max
end

function nn_denormalize_two(x, x_min, x_max) # x ∈ [-1,...,1] -> x ∈ ℜ
    return 0.5*(x_min*(-x).+x_min+x_max.*x.+x_max)
end

ϵ_max = (rand(1)/60)[1]
#ϵ = collect(range(0, ϵ_max, length=timesteps)) # static loading
ϵ_test = collect([range(0, ϵ_max, length=Int(timesteps/3)); range(ϵ_max, -ϵ_max, length=Int(timesteps/3)); range(-ϵ_max, 0, length=Int(timesteps/3))]) # cyclic loading

network(ϵ_test)