## Packages
using LinearAlgebra, StatsBase, Distributions, DataFrames, IterTools, DelimitedFiles, Tidier, ColorSchemes, Plots

"""
Random Jacobian sensu May
"""
function generateWeb(N::Int, C::Float64, σ::Float64)
    ## Create an empty N x N matrix
    J = zeros(N, N)
    ## Randomly assign interaction strengths
    for i ∈ 1:N
        for j ∈ 1:N
            ## Off-diagonals
            if i ≠ j
                ## Assign a random interaction strength between -1 and 1
                J[i, j] = sample([0, 1], pweights([1 - C, C])) * rand(Normal(0, σ))
            ## Diagonals
            elseif i == j
                J[i, j] = -1
            end
        end
    end
    ## Return the generated food web and fix zeros
    J[J .== -0] .= 0
    return J
end

"""
Random web with niche model structure
"""
function generateNicheWeb(S, C, σ::Float64)
    M     = zeros(S, S)
    ci    = zeros(S)
    niche = rand(Float64, S)
    r     = rand(Beta(1, (1 / (2 * C)) - 1), S) .* niche
    for i in 1:S
        ci[i] = rand(Uniform(r[i] / 2, niche[i]))
    end
    ## Set the smallest species niche value to have an n of 0
    min_niche_idx = argmin(niche)
    r[min_niche_idx] = 0.00000001
    ## Inner loop
    for i ∈ 1:S
        for j ∈ 1:S
            ## If the niche of j is in the range of i's niche width
            if niche[j] > (ci[i] - (0.5 * r[i])) && niche[j] < (ci[i] + 0.5 * r[i])
                ## Give the prey a negative value
                M[j, i] += -abs.(rand(Normal(0, σ)))
                ## Give the predator a positive value
                M[i, j] += abs.(rand(Normal(0, σ)))
            end
        end
    end
    ## Diagonals
    M[diagind(M)] .= -1
    ## Output
    return M
end

"""
Make a dummy Laplacian to check component number of a graph
"""
function numComponents(J::Array{Float64})
    ## Make dummy Laplacian
    L  = abs.(J) .* 0
    J₀ = abs.(J)
    J₀[diagind(J₀)] .= 0
    J₀ .= J₀ .+ transpose(J₀)
    L[diagind(L)] .= sum(J₀, dims = 2)
    L .= L .- J₀
    ## Number of zero Laplacian eigenvalues equals our number of components
    nC = sum(round.(abs.(eigen(L).values), digits = 10) .== 0)
    return nC
end

"""
Populate the connectivity matrix
"""
function generateCon(P, q, type)
    ## Setup the array
    N = size(P, 1)
    C = zeros(N, N)
    ## Randomly assign interaction strengths
    for i ∈ 1:N
        for j ∈ 1:N
            ## Off-diagonals
            if i ≠ j
                ## Assign a random dispersal rate with the opposite sign
                if type == "Constrained"
                    C[i, j] = sample([0, -1], pweights([1 - q, q])) * sign(P[i, j]) * rand()
                ## Or a random sign
                elseif type == "Unconstrained"
                    C[i, j] = sample([0, 1], pweights([1 - q, q])) * rand(Uniform(-1, 1))
                end
            ## Diagonals are diffusion rate 1
            elseif i == j
                C[i, j] = 1
            end
        end
    end
    ## Return the array
    C[C .== -0] .= 0
    return C
end

"""
Function that performs sweeps
"""
# Ns = [5]
# Cs = [0.5]
# σs = [0.40]
# qs = [0.95]
# simNum = 1
function sweep(Ns, Cs, σs, qs, simNum, type, wtype)
    ## Make sweep array to fill in
    vecN       = repeat(Ns, simNum)
    sweepArray = collect(product(vecN, Cs, σs, qs))
    total      = length(sweepArray)
    κ          = 0:0.005:10
    nκ         = length(κ)
    res        = zeros(total, 6)
    ## Main loop
    Threads.@threads for i ∈ 1:total
        ## Populate the patch Jacobian
        N₀ = sweepArray[i][1]
        C₀ = sweepArray[i][2]
        σ₀ = sweepArray[i][3]
        q₀ = sweepArray[i][4]
        ## Random matrix
        if wtype == "Random"
            P     = generateWeb(N₀, C₀, σ₀)
        ## Niche model topology
        elseif wtype == "Niche"
            P     = generateNicheWeb(N₀, C₀, σ₀)
        end
        nComp = numComponents(P)
        ## Check for connected web
        while nComp > 1
            ## Random matrix
            if wtype == "Random"
                P     = generateWeb(N₀, C₀, σ₀)
            ## Niche model topology
            elseif wtype == "Niche"
                P     = generateNicheWeb(N₀, C₀, σ₀)
            end
            nComp = numComponents(P)
        end
        ## Populate Connectivity matrix
        C = generateCon(P, q₀, type)
        ## MSF
        S = zeros(nκ)
        for k ∈ 1:nκ
            M    = P .- (κ[k] .* C)
            S[k] = maximum(real.(eigvals(M, scale = false)))
        end
        ## Output
        res[i, 1] = N₀
        res[i, 2] = C₀
        res[i, 3] = σ₀
        res[i, 4] = q₀
        res[i, 5] = maximum(real.(eigvals(P, scale = false))) < 0
        res[i, 6] = res[i, 5] == 1 && any(S .> 0)
    end
    ## Convert to data frame
    d = DataFrame(res, [:N, :C, :σ, :q, :locallyStable, :patternForming])
    ## Return it
    return d
end


## ---
## JIT
## ---
sweep([5], [0.3], [0.2], [0.25], 1, "Constrained", "Random")


## ------------------
## Take it for a spin
## ------------------
@time dat = sweep(10:5:50, [0.25], [0.25], 0:0.1:1, 2500, "Unconstrained", "Niche")

## Summarize outcomes
dat.as = dat.locallyStable .== 1 .&& dat.patternForming .== 0
dat.ns = dat.locallyStable .== 0
dat.pf = dat.patternForming .== 1

## Summary data
sd = @chain dat begin
    @group_by(N, q)
    @summarize(mas = mean(as),
               mns = mean(ns),
               mpf = mean(pf))
    @ungroup
    @arrange(N, q)
end

## Look
println(sd)

## 2D line plot
# plot(sd.q, sd.mpf, legend = :none)
# scatter!(sd.q, sd.mpf)

## Plots
surface(sd.N, sd.q, sd.mpf,
        tickfont = font(8, "Computer Modern"), 
        guidefont = font(10, "Computer Modern"),
        xlabel = "N", ylabel = "q", zlabel = "Prop. pattern forming webs",
        alpha = 1, legend = :none,
        widen = true,
        zlim = (0, 1),
        c = :seaborn_icefire_gradient,
        xtickfonthalign = :left, xtickfontvalign = :top,
        ytickfonthalign = :center, ytickfontvalign = :top,
        ztickfonthalign = :center, ztickfontvalign = :bottom,
        size = (300, 300))

## Save figure
gr(size = (300, 300), dpi = 480)
savefig("~/Desktop/nicheUnconstrained.png")

## Save data
writedlm("nicheUnconstrained.csv", Iterators.flatten(([names(dat)], eachrow(dat))), ',')



