using SparseArrays
using Random


function Forest(g::Graph,k, l=4000)
    w, t = compute_forest(g, l)

    return partialsortperm(w .* (1 .- g.s), 1:k, rev=true), t
end

"""
    compute_forest(g::Graph, epsilon)

Estimate node visit probabilities using random forest sampling.
Returns (z, time, k) where:
- z: probability vector
- time: execution time in seconds
- k: number of samples
"""
function compute_forest(g::Graph, k)
    
    w = @timed begin
        visits = zeros(g.n)
        @inbounds for _ in 1:k
            root = random_forest(g)
            for j in root
                j > 0 && (visits[j] += 1)
            end
        end
        visits ./= k
    end
    
    return w.value, w.time
end

"""
    random_forest(g::Graph, indices, indptr)

Generate a random spanning forest using depth-limited random walks.
Returns root vector where root[i] is the root of node i's tree.
"""
function random_forest(g::Graph)
    in_forest = falses(g.n)
    next = fill(-1, g.n)
    root = fill(-1, g.n)

    @inbounds for i in 1:g.n
        u = i
        while !in_forest[u]
            d = g.d[u]
            if rand() ≤ g.f[u] || d == 0  # Absorption or isolated node
                in_forest[u] = true
                root[u] = d == 0 ? -1 : u
            else
                neighbors = @view g._colval[g._rowptr[u]:(g._rowptr[u+1]-1)]
                next[u] = neighbors[rand(1:d)]
                u = next[u]
            end
        end
        
        # Path compression
        r = root[u]
        u = i
        while !in_forest[u]
            in_forest[u] = true
            root[u] = r
            u = next[u]
        end
    end
    return root
end

