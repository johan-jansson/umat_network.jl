using Flux, CUDA                      # https://fluxml.ai/Flux.jl/stable/models/basics/ & https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl
using Plots
plotly()
theme(:dark)

function mixed_hardening(ⁿσ, ⁿα, ⁿκ, Δϵ, E, H, σʸ, r)
    σₜ = ⁿσ + E * Δϵ                                                            # trial stress
    σₜʳᵉᵈ = σₜ - ⁿα                                                             # reduced elastic stress
    Φₜ = abs(σₜʳᵉᵈ) - (σʸ + ⁿκ)                                                 # trial yield function
    if Φₜ > 0                                                                   # plastic loading
        μ = Φₜ / (E + H)                                                        # plastic multiplier
        Eₜ = (E * H) / (E + H)                                                  # tangent stiffness
    else                                                                        # elastic loading
        μ = 0                                                                   # plastic multiplier
        Eₜ = E                                                                  # tangent stiffness
    end
    σ = σₜ - μ * E * sign(σₜʳᵉᵈ)                                                # integrated continuum stress
    κ = ⁿκ + μ * r * H                                                          # integrated micro-stress
    α = ⁿα + (1 - r) * H * μ * sign(σₜʳᵉᵈ)                                      # integrated back-stress
    return σ, α, κ, Eₜ
end
function loadsteps(timesteps, models, E, H, σʸ, r)
    α = zeros(timesteps)
    σ = zeros(timesteps)
    κ = zeros(timesteps)
    σ_tot = zeros(timesteps, models)
    ϵ_tot = zeros(timesteps, models)
    for num ∈ 1:models
        ϵ_max = (rand(1)/60)[1]
        ϵ = collect(range(0, ϵ_max, length=timesteps))
        #ϵ = collect([range(0, ϵ_max, length=Int(timesteps/3)); range(ϵ_max, -ϵ_max, length=Int(timesteps/3)); range(-ϵ_max, 0, length=Int(timesteps/3))])
        Δϵ = diff(ϵ, dims=1)
        for timestep ∈ 1:timesteps
            if timestep == timesteps break end
            σ[timestep+1], α[timestep+1], κ[timestep+1], Eₜ = mixed_hardening(σ[timestep], α[timestep], κ[timestep], Δϵ[timestep], E, H, σʸ, r)
        end
        ϵ_tot[:,num] = ϵ
        σ_tot[:,num] = σ
    end
    return Float32.(ϵ_tot), Float32.(σ_tot)
end
function nn_normalize_one(x) # x ∈ ℜ -> x ∈ [0,...,1]
    x_min = findmin(x)[1]
    x_max = findmax(x)[1]
    return ((x .- x_min)/(x_max - x_min)), x_min, x_max
end
function nn_denormalize_one(x, x_min, x_max) # x ∈ [0,...,1] -> x ∈ ℜ
    return (x.*(x_max - x_min) .+ x_min)
end
function nn_normalize_two(x) # x ∈ ℜ -> x ∈ [-1,...,1]
    x_min = findmin(x)[1]
    x_max = findmax(x)[1]
    return ((x .- ((x_min+x_max)/2))/((x_max-x_min)/2)), x_min, x_max
end
function nn_denormalize_two(x, x_min, x_max) # x ∈ [-1,...,1] -> x ∈ ℜ
    return 0.5*(x_min*(-x).+x_min+x_max.*x.+x_max)
end
# function standardize_statsbase() # standardize data ("whitening") (x ∈ [-1.486..., ..., 1.496...] with σ = 1 and μ = 0)
#     σ = nn_denormalize_two(σ_train, σ_train_min, σ_train_max)
#     ϵ = nn_denormalize_two(ϵ_train, ϵ_train_min, ϵ_train_max)
#     # standardize
#     using StatsBase
#     ϵ_transform = StatsBase.fit(ZScoreTransform, ϵ, dims=1)
#     ϵ_std = StatsBase.transform(ϵ_transform, ϵ)
#     # destandardize
#     StatsBase.reconstruct(ϵ_transform, ϵ_std)
# end
# function normalize_statsbase()
#     σ = nn_denormalize_two(σ_train, σ_train_min, σ_train_max)
#     ϵ = nn_denormalize_two(ϵ_train, ϵ_train_min, ϵ_train_max)
#     # normalize
#     using StatsBase
#     ϵ_transform = StatsBase.fit(UnitRangeTransform, ϵ, dims=1)
#     ϵ_norm = StatsBase.transform(ϵ_transform, ϵ)
#     # denormalize
#     StatsBase.reconstruct(ϵ_transform, ϵ_norm)
# end

# settings
timesteps = 10
models = 1000
percentage_train = 0.5

# generate data
ϵ, σ = loadsteps(timesteps, models, 70000, 20000, 100, 0)
ϵ_train = ϵ[:, 1:Int(models*percentage_train)]
σ_train = σ[:, 1:Int(models*percentage_train)]
ϵ_test = ϵ[:, Int(models*percentage_train)+1:end]
σ_test = σ[:, Int(models*percentage_train)+1:end]

# pre-conditioning
# normalize
ϵ_train, ϵ_train_min, ϵ_train_max = nn_normalize_two(ϵ_train)
σ_train, σ_train_min, σ_train_max = nn_normalize_two(σ_train)
ϵ_test, ϵ_test_min, ϵ_test_max = nn_normalize_two(ϵ_test)
σ_test, σ_test_min, σ_test_max = nn_normalize_two(σ_test)
# vectorize
train_data = Flux.Data.DataLoader(ϵ_train, σ_train)
#test_data = Flux.Data.DataLoader(ϵ_test, σ_test)

# deep recurrent neural network
network = Chain(LSTM(timesteps, timesteps),Dense(timesteps, timesteps))

# L2-norm objective function ("loss function")
obj(x, y) = Flux.Losses.mse(network(x), y)

# load network
#BSON.@load "cyclic-10000-nn_normalize_two.bson" network

# training
function evalcb()
    @show(obj(ϵ_train,σ_train))
    if obj(ϵ_train,σ_train) < 1e-4 
        println("Epoch termination criteria met!")
        Flux.stop() 
    end
end

@time Flux.@epochs 1000 Flux.train!(obj, params(network), train_data, ADAM(0.001), cb = Flux.throttle(evalcb, 1))

## evaluate network performance manually
plot(legend=:bottomright, xlabel="strain [-]", ylabel="stress [MPa]", size=(1000,750))
plot!(nn_denormalize_two(ϵ_test[:,1], ϵ_train_min, ϵ_train_max), nn_denormalize_two(σ_test[:,1], σ_train_min, σ_train_max), color=:red, ls=:solid, lw=3, label="constitutive model #1")
plot!(nn_denormalize_two(ϵ_test[:,2], ϵ_train_min, ϵ_train_max), nn_denormalize_two(σ_test[:,2], σ_train_min, σ_train_max), color=:blue, ls=:solid, lw=3, label="constitutive model #2")
plot!(nn_denormalize_two(ϵ_test[:,3], ϵ_train_min, ϵ_train_max), nn_denormalize_two(σ_test[:,3], σ_train_min, σ_train_max), color=:green, ls=:solid, lw=3, label="constitutive model #3")
plot!(nn_denormalize_two(ϵ_test[:,4], ϵ_train_min, ϵ_train_max), nn_denormalize_two(σ_test[:,4], σ_train_min, σ_train_max), color=:yellow, ls=:solid, lw=3, label="constitutive model #4")
plot!(nn_denormalize_two(ϵ_test[:,5], ϵ_train_min, ϵ_train_max), nn_denormalize_two(σ_test[:,5], σ_train_min, σ_train_max), color=:white, ls=:solid, lw=3, label="constitutive model #5")
plot!(nn_denormalize_two(ϵ_test[:,1], ϵ_train_min, ϵ_train_max), nn_denormalize_two(network(ϵ_test[:,1]), σ_train_min, σ_train_max)[:,1], color=:red, ls=:dash, lw=2, label="network prediction #1")
plot!(nn_denormalize_two(ϵ_test[:,2], ϵ_train_min, ϵ_train_max), nn_denormalize_two(network(ϵ_test[:,2]), σ_train_min, σ_train_max)[:,2], color=:blue, ls=:dash, lw=2, label="network prediction #2")
plot!(nn_denormalize_two(ϵ_test[:,3], ϵ_train_min, ϵ_train_max), nn_denormalize_two(network(ϵ_test[:,3]), σ_train_min, σ_train_max)[:,3], color=:green, ls=:dash, lw=2, label="network prediction #3")
plot!(nn_denormalize_two(ϵ_test[:,4], ϵ_train_min, ϵ_train_max), nn_denormalize_two(network(ϵ_test[:,4]), σ_train_min, σ_train_max)[:,4], color=:yellow, ls=:dash, lw=2, label="network prediction #4")
plot!(nn_denormalize_two(ϵ_test[:,5], ϵ_train_min, ϵ_train_max), nn_denormalize_two(network(ϵ_test[:,5]), σ_train_min, σ_train_max)[:,5], color=:white, ls=:dash, lw=2, label="network prediction #5")
gui()

# save network
BSON.@save "network.bson" network