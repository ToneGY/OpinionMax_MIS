

A Julia implementation of various influence maximization algorithms with ground truth computation and evaluation metrics. 

This is the code repository for the Neurips 2025 paper *Opinion Maximization in Social Networks by Modifying Internal Opinions*. 

## Features

- Three influence maximization algorithms:
  - MIS (Maximal Influence Selection)
  - RWB (Random Walk Based)
  - Forest Fire Sampling
- Ground truth computation
- Multiple resistance coefficient distributions:
  - Uniform (`uni`)
  - Exponential (`exp`)
  - Power Law (`pow`)
- Evaluation metrics:
  - Precision
  - NDCG
  - Overall Opinion

## Installation

1. **Install Julia** (version 1.6 or higher recommended):
   - Download from [Julia's official website](https://julialang.org/downloads/)
   - Or use a package manager:
     ```bash
     # For Ubuntu/Debian
     sudo apt-get install julia
     
     # For macOS with Homebrew
     brew install julia
     ```

2. **​​Install required Julia packages**:
   - Launch Julia and install dependencies:
        ```julia
        # Enter package mode of julia by pressing ]
        pkg> add DelimitedFiles
        pkg> add LinearAlgebra
        pkg> add SparseArrays
        pkg> add DataStructures
        pkg> add Random
        pkg> add Distributions
        ```
## Usage
#### 1. Generate Ground Truth
First generate ground truth files for your graph:
```bash
julia ground_truth.jl <graph_name> <distribution>
```
Where:
- `<graph_name>`: Name of the graph in `dataInfo.jl`
- `<distribution>`: One of `uni`, `exp`, or `pow`

Example:
```bash
julia ground_truth.jl hamster uni
```

#### 2. Run Influence Maximization
```bash
julia main.jl <graph_name> <distribution> <algorithm> <k> [<optional_param>]
```
Parameters:
- `<graph_name>`: Name of the graph
- `<distribution>`: Distribution type (`uni`, `exp`, or `pow`)
- `<algorithm>`: One of `mis`, `rwb`, or `forest`
- `<k>`: Number of seed nodes (must be > 0 and < total nodes)
- `<optional_param>`:
  - For rwb: Error bound ε (0 < ε < 1)
  - For forest: Sample count (> 0)
  - Not needed for mis

Examples:
```bash
# Run MIS with k=50
julia main.jl hamster uni mis 50

# Run RWB with ε=0.1 and k=50
julia main.jl hamster uni rwb 50 0.1

# Run Forest with 4000 samples and k=50
julia main.jl hamster uni forest 50 4000
```

## Available Graphs
The following graphs are included in the dataset:

- hamster
- DBLP
- Google

To add a new graph:
1. Place the graph file in the `data` directory
2. Update `datainfo.jl` to include the new graph

## Output
The program outputs:

- Execution time
- Precision score
- NDCG score
- Overall opinion score

Example output:

┌ Info: Results

│   algorithm = "forest"

│   time = 0.2330849

│   precision = 0.99

│   ndcg = 0.999909696019612

└   overall_opinion = 1429.8283731658878
