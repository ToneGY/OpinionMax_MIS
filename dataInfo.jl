struct DataInfo
    filename::String
    n::Int32
    m::Int32
    skipline::Int8
    is_direct::Bool
end

const DATASET_INFO = Dict{String,DataInfo}(
    "hamster" => DataInfo("data/hamster.mtx", 2426, 16630, 1, false),
    "DBLP" => DataInfo("data/dblp.mtx", 317080, 1049866, 0, false),
    "Google" => DataInfo("data/google.mtx", 875713, 5105039, 4, true),
)

get_data() = DATASET_INFO