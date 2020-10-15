# Recurrent neural network for 1D mixed kinematic hardening plasticity subject to cyclic loading
# Chain(LSTM(timesteps, timesteps*2),Dense(timesteps*2, timesteps)) with loss: 2e-5 after 10 epochs
# 9000 models, 90 timesteps, E = 70000,  H = 20000,  σʸ = 100, r = 0.5
using Flux, BSON

cd(@__DIR__)

BSON.@load "network.bson" network
σ_train_min = -150
σ_train_max = 150
println("hello world, network.bson load successfully!")

function nn_normalize_two(x) # x ∈ ℜ -> x ∈ [-1,...,1]
    x_min = findmin(x)[1]
    x_max = findmax(x)[1]
    return ((x .- ((x_min+x_max)/2))/((x_max-x_min)/2)), x_min, x_max
end

function nn_denormalize_two(x, x_min, x_max) # x ∈ [-1,...,1] -> x ∈ ℜ
    return 0.5*(x_min*(-x).+x_min+x_max.*x.+x_max)
end

ϵ_max = (rand(1)/60)[1]
ϵ = collect([range(0, ϵ_max, length=Int(90/3)); range(ϵ_max, -ϵ_max, length=Int(90/3)); range(-ϵ_max, 0, length=Int(90/3))])
ϵ, x_min, x_max = nn_normalize_two(ϵ)
σ = nn_denormalize_two(network(ϵ), σ_train_min, σ_train_max)[:,1]
println("\ntest ϵ: $ϵ\n\nnetwork σ: $σ")
