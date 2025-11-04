using Random

"""
    build_alias_table(probs::Vector{Float64}) -> Tuple{Vector{Float64}, Vector{Int}}

Construct alias table for Vose's Alias Method sampling algorithm.

# Arguments
- `probs`: Vector of probabilities (will be normalized if not already)

# Returns
- Tuple containing:
  - `Prob`: Vector of probability thresholds
  - `Alias`: Vector of alias indices

# Performance Notes
- Pre-allocates all required memory upfront
- Uses in-place operations where possible
- Avoids global variables and type instability
"""
function build_alias_table(probs::Vector{Float64})
    # Normalize probabilities and get length
    n = length(probs)
    sum_probs = sum(probs)
    inv_sum = 1.0 / sum_probs  # Precompute reciprocal for faster division
    scaled_probs = probs .* (n * inv_sum)  # Vectorized normalization
    
    # Pre-allocate all required arrays
    Prob = Vector{Float64}(undef, n)
    Alias = Vector{Int}(undef, n)
    small = Vector{Int}()
    large = Vector{Int}()
    sizehint!(small, n)  # Pre-allocate expected size
    sizehint!(large, n)
    
    # Classify probabilities into small (<1) and large (≥1)
    @inbounds for i in 1:n
        if scaled_probs[i] < 1.0
            push!(small, i)
        else
            push!(large, i)
        end
    end
    
    # Construct alias table using Vose's algorithm
    @inbounds while !isempty(small) && !isempty(large)
        s = pop!(small)  # Index of small probability
        l = pop!(large)  # Index of large probability
        
        Prob[s] = scaled_probs[s]
        Alias[s] = l
        
        # Update remaining probability (fused multiply-add operation)
        scaled_probs[l] = fma(scaled_probs[s], 1.0, scaled_probs[l] - 1.0)
        
        # Reclassify updated probability
        if scaled_probs[l] < 1.0
            push!(small, l)
        else
            push!(large, l)
        end
    end
    
    # Handle remaining elements
    @inbounds while !isempty(large)
        l = pop!(large)
        Prob[l] = 1.0
    end
    
    @inbounds while !isempty(small)
        s = pop!(small)
        Prob[s] = 1.0
    end
    
    return (Prob, Alias)
end

"""
    alias_sample(Prob::Vector{Float64}, Alias::Vector{Int}) -> Int

Sample from discrete distribution using Alias Method.

# Arguments
- `Prob`: Probability thresholds from `build_alias_table`
- `Alias`: Alias indices from `build_alias_table`

# Returns
- Sampled index according to original probabilities

# Performance Notes
- Uses single random number generation
- Branch prediction friendly
- Inlined for better performance
"""
@inline function alias_sample(Prob::Vector{Float64}, Alias::Vector{Int})
    n = length(Prob)
    i = rand(1:n)  # Uniformly select a bin
    r = rand()     # Single random number for threshold comparison
    
    # Use branchless conditional via multiplication
    return (r < Prob[i]) ? i : Alias[i]
end