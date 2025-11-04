using DelimitedFiles
include("tools/graph.jl")
include("dataInfo.jl")
include("alg/mis.jl")

function compute_exact(g::Graph)
    T = I - Diagonal( (1 .- g.f) ./ g.d ) * g.A
    w = vec(sum( inv(Matrix(T))*Diagonal(g.f), dims=1))
    return w
end

function compute(name, distr)
    data = get_data()
    
    # Map distr argument to the appropriate distribution type
    distr_type = if distr == "uni"
        TruncatedUniform()
    elseif distr == "exp"
        ScaledExponential()
    elseif distr == "pow"
        ScaledPowerLaw()
    else
        error("Invalid distribution type: $distr. Must be one of: uni, exp, pow")
    end
    
    g = load_graph(data[name], Uniform(), distr_type)
    writedlm("ground_truth/$(name)_$(distr)_f.txt", g.f)
    writedlm("ground_truth/$(name)_$(distr)_s.txt", g.s)
    # w = compute_exact(g)
    w,_ = globalInfApprox(g, 1e-12)
    writedlm("ground_truth/$(name)_$(distr).txt", w)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        error("Usage: julia script.jl name distr\nwhere distr is one of: uni, exp, pow")
    end
    
    name = ARGS[1]
    distr = ARGS[2]
    
    compute(name, distr)
end