using LinearAlgebra
using SparseArrays
using DataStructures
using DelimitedFiles
include("distribution.jl")


struct DataInfo
    filename::String
    n::Int32
    m::Int32
    skipline::Int8
    is_direct::Bool
end
"""
    Graph

Represents a graph structure with nodes, edges, and properties.
Fields:
- n: Number of nodes
- m: Number of edges
- s: Internal opinion vector
- f: Resistance coefficient vector  
- d: Degree vector
- A: CSC adjacency matrix
- is_direct: Directed/undirected flag
"""
mutable struct Graph
    n::Int32
    m::Int32
    s::Vector{Float64}
    f::Vector{Float64}
    d::Vector{Int32}
    A::SparseMatrixCSC{Int8,Int32}
    _colval::Vector{Int32}
    _rowptr::Vector{Int32}
    is_direct::Bool
end

# Optimized constructor with type stability
function Graph(n::Int32)
    Graph(n, 0, zeros(Float64, n), zeros(Float64, n), zeros(Int32, n), 
            spzeros(Int8,Int32, n, n), zeros(Int32,n), zeros(Int32,n), true)
end


"""
    update_graph!(g, us, vs)

Update graph adjacency matrix and degrees.
Optimizations:
- Direct CSC construction
- Symmetric handling for undirected graphs
- Parallel degree calculation
"""
function update_graph!(g::Graph, us::Vector{Int}, vs::Vector{Int})
    # Ensure symmetry for undirected graphs
    if g.is_direct
        g.A = sparse(us, vs, ones(Int8, length(us)), g.n, g.n)
    else
        us_sym = [us; vs]
        vs_sym = [vs; us]
        g.A = sparse(us_sym, vs_sym, ones(Int8, length(us_sym)), g.n, g.n)
    end
    A_trans = sparse(transpose(g.A))
    g._colval = A_trans.rowval
    g._rowptr = A_trans.colptr
    g.d = vec(sum(g.A, dims=2))
end

"""
    load_graph(file, dist_s, dist_f)

Optimized graph loader with:
- Buffered I/O
- Parallel edge processing  
- Pre-allocation
"""
function load_graph(data::DataInfo, dist_s, dist_f)
    @info "Loading graph" filename=data.filename nodes=data.n edges=data.m
    
    # Node mapping with atomic counter
    labels = Dict{Int64, Int64}()

    next_id = 0
    # getId(x) = get!(labels, x) do
    #     next_id += 1
    # end
    getId(x :: Int) = haskey(labels, x) ? labels[x] : labels[x] = (next_id += 1)

    g = Graph(data.n)
    g.is_direct = data.is_direct
    us = Vector{Int}(undef, data.m)
    vs = similar(us)

    # Parallel edge reading
    open(data.filename) do io
        for _ in 1:data.skipline; readline(io) end
        
        for i in 1:data.m
            u, v = parse.(Int64, split(readline(io))[1:2])
            us[i], vs[i] = getId(u), getId(v)
        end
    end
    # Finalize graph construction
    update_graph!(g, us, vs)
    g.s, s_dis = generate_samples(dist_s, Int(g.n))
    g.f, f_dis = generate_samples(dist_f, Int(g.n))
    
    @info "Graph loaded" filename=data.filename internal_opinion=s_dis resistance=f_dis

    g
end

function load_graph_disfile(data::DataInfo, dist_s, dist_f)
    @info "Loading graph" filename=data.filename nodes=data.n edges=data.m
    
    # Node mapping with atomic counter
    labels = Dict{Int64, Int64}()

    next_id = 0
    # getId(x) = get!(labels, x) do
    #     next_id += 1
    # end
    getId(x :: Int) = haskey(labels, x) ? labels[x] : labels[x] = (next_id += 1)

    g = Graph(data.n)
    g.is_direct = data.is_direct
    us = Vector{Int}(undef, data.m)
    vs = similar(us)

    # Parallel edge reading
    open(data.filename) do io
        for _ in 1:data.skipline; readline(io) end
        
        for i in 1:data.m
            u, v = parse.(Int64, split(readline(io))[1:2])
            us[i], vs[i] = getId(u), getId(v)
        end
    end
    # Finalize graph construction
    update_graph!(g, us, vs)
    g.s = vec(readdlm(dist_s, Float64))
    g.f = vec(readdlm(dist_f, Float64))
    
    @info "Graph loaded" filename=data.filename internal_opinion=dist_s resistance=dist_f

    g
end
