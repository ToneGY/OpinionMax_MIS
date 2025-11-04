using LinearAlgebra
using SparseArrays
include("../tools/circularBufferQueue.jl")
"""
    compute_topK(g::Graph, k::Int)

Compute top-K nodes using push-based algorithm with adaptive precision control.
Returns indices of top-K nodes based on growth metric z.*(1 .- g.s)
"""
function maxInfluenceSeletor(g::Graph, k::Int, epsilon=1e-3)
    result = nothing
    mis = @timed begin
        # Initial push phase
        w, ra= globalInfApprox(g, epsilon)
        growth = w .* (1 .- g.s)
        
        # Initial candidate selection
        topK, cans, restK = _select_candidates(growth, k, epsilon)
        if restK <= 0 
            result = topK
        else
            # Refinement phase with adaptive precision
            dict = Dict{Int,Vector{Float64}}()
            for c in cans
                r = zeros(g.n)
                r[c] = g.f[c] * (1 - g.s[c])
                dict[c] = r
            end

            nn = 1
            while true
                epsilon = 1.0 / nn
                nn *= 2
                nz = Float64[]
                for c in cans
                    Δ, r = targetedNodeRefine(g, ra, dict[c], epsilon)
                    dict[c] = r
                    growth[c] += Δ
                    push!(nz, growth[c])
                end
                
                new_topK, new_cans, restK = _select_candidates(nz, restK, epsilon, false)
                append!(topK, cans[new_topK])
                if restK <= 0 
                    result = topK
                    break
                end
                cans = cans[new_cans]
            end
        end
    end
    return result, mis.time
end

"""
    globalInfApprox(indices, indptr, d, f, s, epsilon)

Core push algorithm implementation with queue-based propagation.
Returns (z, r, num_operations) where:
- z: accumulated values
- r: residual values
- num_operations: count of operations performed
"""
function globalInfApprox(g::Graph, epsilon::Float64)
    n = g.n
    r_max = fill(epsilon, n)
    w = zeros(n)
    r = ones(n)
    queue = _CircularBufferQueue(n)
    mark = trues(n)
    for i::Int32 in 1:n
        push_q!(queue, i)
    end

    while !isempty_q(queue)
        i = pop_q!(queue)
        mark[i] = false
        rv = g.f[i] * r[i]
        ri = r[i]
        @inbounds w[i] += rv
        r[i] = 0
        
        @views neighbors = g._colval[g._rowptr[i]:(g._rowptr[i+1]-1)]
        @simd for j in neighbors
            @inbounds r[j] += (ri- rv) / g.d[i]
            @inbounds if r[j] > r_max[j] && !mark[j]
                push_q!(queue, j)
                mark[j] = true
            end
        end
    end
    
    return w, r
end

"""
    targetedNodeRefine(g, r, epsilon)

Refinement push step for candidate nodes.
Returns (delta_w, updated_r) with improved precision.
"""
function targetedNodeRefine(g::Graph, ra::Vector{Float64}, r::Vector{Float64}, epsilon::Float64)
    n = g.n
    epsilon /= sum(r)
    r_max = epsilon .* g.f
    delta_w = 0.0
    queue = _CircularBufferQueue(n)
    mark = trues(n)
    
    for i::Int32 in 1:n
        push_q!(queue, i)
    end

    while !isempty_q(queue)
        i = pop_q!(queue)
        mark[i] = false
        rv = r[i]
        
        @inbounds delta_w += ra[i] * rv
        r[i] = 0
        
        @views neighbors = g.A.rowval[g.A.colptr[i]:(g.A.colptr[i+1]-1)]
        @simd for j in neighbors
            @inbounds r[j] += (1 - g.f[j]) * rv / g.d[j]
            @inbounds if r[j] > r_max[j] && !mark[j]
                push_q!(queue, j)
                mark[j] = true
            end
        end
    end
    
    return delta_w, r
end

"""
    _select_candidates(vec, k, epsilon, relative=true)

Select top-K candidates with epsilon tolerance.
Returns (topK, candidates, remaining) where:
- topK: definitely in top-K
- candidates: potential candidates
- remaining: number left to select
"""
function _select_candidates(vec::AbstractVector, k::Int, 
                          epsilon::Float64, relative=true)
    kv = partialsort(copy(vec), k:(k+1), rev=true)
    
    bounds = relative ? 
        ((1-epsilon)*kv[1], kv[2]/(1-epsilon)) : 
        (kv[1]-epsilon, kv[2]+epsilon)
    
    possible = findall(x -> x > bounds[1], vec)
    length(possible) == k && return (sort(possible, by = i -> vec[i], rev = true), Int[], 0)
    
    topK = Int[]
    candidates = Int[]
    
    for i in possible
        vec[i] >= bounds[2] ? push!(topK, i) : push!(candidates, i)
    end
    
    return (sort(topK, by = i -> vec[i], rev = true), candidates, k - length(topK))
end