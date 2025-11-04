include("../tools/aliasTable.jl")


function RWB(g::Graph, k, epsilon)
    w, t = compute_rw_random(g, epsilon)
    return partialsortperm(w .* (1 .- g.s), 1:k, rev=true), t
end

"""
    compute_rw_random(g::Graph, epsilon, k=0, o=missing)

Perform random walks on graph `g` with precision control `epsilon`.
Returns (w, t, 0) where:
- w: visit probability vector
- t: execution time
"""
function compute_rw_random(g::Graph, epsilon, k=0, o=missing)
    # Initialize parameters
    o = ismissing(o) ? ones(g.n) : o
    sum_v = sum(o)
    k = epsilon ≠ 0 ? ceil(Int, g.n * log(g.n) / epsilon^2) : k
    
    # Precompute sampling tables
    Prob, Alias = build_alias_table(o)
    
    # Main computation
    w = @timed begin
        visits = zeros(g.n)
        for _ in 1:k
            start = alias_sample(Prob, Alias)
            target = _random_walk(g._colval, g._rowptr, g.f, start)
            (target == -1) && continue
            visits[target] += 1
        end
        visits ./= k / sum_v
    end
    
    return w.value, w.time
end

"""
    _random_walk(indices, indptr, f, start)

Single random walk with absorption probabilities f.
Returns terminating node index.
"""
function _random_walk(indices::Vector{Int32}, indptr::Vector{Int32}, 
                     f::Vector{Float64}, start::Int)
    current = start
    while true
        rand() ≤ f[current] && return current  # Absorption
        
        # Get neighbors from CSR format
        neighbors = @view indices[indptr[current]:(indptr[current+1]-1)]
        (length(neighbors) == 0) && return -1
        current = neighbors[rand(1:length(neighbors))]  # Uniform neighbor selection
    end
end
