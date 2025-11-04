include("graph.jl")

function getPrecision(g::Graph, true_value, setK)
    k = length(setK)
    Δ = true_value .* (1 .- g.s)
    true_sizeK = partialsortperm(Δ, 1:k, rev=true)
    length(intersect(true_sizeK, setK))/k
end

function getNdcg(g::Graph, true_value, setK)
    Δ = true_value .* (1 .- g.s)
    k = length(setK)
    dcg = 0.0
    for (rank, idx) in enumerate(setK)
        relevance = Δ[idx]  
        dcg += (relevance) / log2(rank + 1)  
    end
    
    # compute IDCG (Ideal DCG)
    ideal_ranking = partialsort(Δ, 1:k, rev=true)
    idcg = 0.0
    for (rank, relevance) in enumerate(ideal_ranking)
        idcg += (relevance) / log2(rank + 1)
    end
    
    idcg == 0 ? 0.0 : dcg / idcg
end

function getOverallOpinion(g::Graph, true_value, setK)
    s = copy(g.s)
    s[setK] .= 1
    sum(true_value .* s)
end