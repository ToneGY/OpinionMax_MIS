include("tools/graph.jl")
include("datainfo.jl")
include("alg/rwb.jl")
include("alg/forest.jl")
include("alg/mis.jl")
include("tools/metrics.jl")

function check_data_files(name, distr)
    data = get_data()
    
    # Check if name exists in data
    if !haskey(data, name)
        available_names = join(keys(data), ", ")
        error("""
        Graph name '$name' not found in dataset.
        Available options are: $available_names
        Please add the graph file or choose from the available names.
        """)
    end
    
    # Check if ground truth files exist
    required_files = [
        "ground_truth/$(name)_$(distr)_s.txt",
        "ground_truth/$(name)_$(distr)_f.txt",
        "ground_truth/$(name)_$(distr).txt"
    ]
    
    missing_files = [f for f in required_files if !isfile(f)]
    
    if !isempty(missing_files)
        error("""
        Missing ground truth files for $name with $distr distribution:
        $(join(missing_files, "\n"))
        
        Please first generate these files using ground_truth.jl:
        julia ground_truth.jl $name $distr
        """)
    end
end

function main()
    # Check minimum number of arguments
    if length(ARGS) < 4
        error("""
        Usage: julia main.jl name distr algorithm k [optional_param]
        Where:
        - name: graph name (e.g., "hamster")
        - distr: distribution type ("uni", "exp", or "pow")
        - algorithm: "mis", "forest", or "rwb"
        - k: number of seeds (0 < k < n)
        - optional_param: depends on algorithm:
          * mis: none
          * rwb: error bound (0 < ϵ < 1)
          * forest: sample count (> 0)
        """)
    end

    # Parse required arguments
    name = ARGS[1]
    distr = ARGS[2]
    algorithm = ARGS[3]
    k_str = ARGS[4]

    # Validate distribution type
    if !(distr in ["uni", "exp", "pow"])
        error("Invalid distribution type: $distr. Must be one of: uni, exp, pow")
    end

    # Validate algorithm
    if !(algorithm in ["mis", "forest", "rwb"])
        error("Invalid algorithm: $algorithm. Must be one of: mis, forest, rwb")
    end

    # Check data files exist
    check_data_files(name, distr)

    # Load graph and validate k
    data = get_data()
    g = load_graph_disfile(data[name], "ground_truth/$(name)_$(distr)_s.txt", "ground_truth/$(name)_$(distr)_f.txt")
    
    try
        k = parse(Int, k_str)
        if k <= 0 || k >= g.n
            error("k must be > 0 and < $(g.n), got $k")
        end
    catch e
        error("Invalid k value: $k_str. Must be an integer > 0 and < $(g.n)")
    end
    k = parse(Int, k_str)

    # Load ground truth
    true_value = vec(readdlm("ground_truth/$(name)_$(distr).txt", Float64))

    # Execute selected algorithm with parameter validation
    if algorithm == "mis"
        if length(ARGS) > 4
            @warn "MIS doesn't take optional parameters, ignoring extra arguments"
        end
        
        @info "Running MIS with k=$k"
        v, t = maxInfluenceSeletor(g, k)
        
    elseif algorithm == "rwb"
        ϵ  = 0
        if length(ARGS) < 5
            ϵ = 0.1
        else
            ϵ = parse(Float64, ARGS[5])
            if ϵ <= 0 || ϵ >= 1
                error("RWB error bound must be > 0 and < 1, got $ϵ")
            end
        end
        @info "Running RWB with k=$k, ϵ=$ϵ, $(ceil(Int, g.n * log(g.n) / ϵ^2)) samples"
        rwb, t = RWB(g, k, ϵ)
        v = rwb  # Assuming RWB returns the seed set
        
    elseif algorithm == "forest"
        sample_count = 0
        if length(ARGS) < 5
            sample_count = 4000
        else
            sample_count = parse(Int, ARGS[5])
            if sample_count <= 0
                error("Forest sample count must be > 0, got $sample_count")
            end
        end      

        @info "Running Forest with k=$k, samples=$sample_count"
        forest, t = Forest(g, k, sample_count)
        v = forest  # Assuming Forest returns the seed set
    end

    # Compute metrics
    precision = getPrecision(g, true_value, v)
    ndcg = getNdcg(g, true_value, v)
    overall_opinion = getOverallOpinion(g, true_value, v)
    
    @info "Results" algorithm=algorithm time=t precision=precision ndcg=ndcg overall_opinion=overall_opinion
end


# Execute main function if run as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end