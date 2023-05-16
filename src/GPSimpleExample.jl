module GPSimpleExample
    using Arrow
    using DataFrames
    using Proj
    using Statistics
    using GaussianProcesses
    using Interpolations
    using CairoMakie
    
    # Write your package code here.

    """
        bin(x::Vector{<:Number}, y::Array{<:Number}, xbin_edges::Union{Vector{<:Number},StepRangeLen{}}; method::F = mean) where {F<:Function}

    Fast binning of `y` as a function of `x`. Returns `x_binned`, `y_binned`, `bin_count`. `method` 
    specifies approached used for aggregating binned values. NOTE: `x_binned` == 0 and 
    `y_binned` == 0 when `bin_count` == 0.

    # see https://discourse.julialang.org/t/performance-optimization-of-a-custom-binning-funciton/91616
    """
    function bin(
        x::Vector{<:Number}, 
        y::Array{<:Number}, 
        xbin_edges::Union{Vector{<:Number},StepRange{}};
        method::F = mean,
        ) where {F<:Function}

        # find bin breakpoints
        p = sortperm(vcat(x, xbin_edges))
        bins = findall(>(length(x)), p)

        # initialize outputs
        bin_count = Int.(diff(bins).-1)
        x_binned = zeros(eltype(x), length(bin_count))
        y_binned = zeros(eltype(y), length(bin_count), size(y, 2))

        # calculate binned metrics
        for i = findall(bin_count .> 0)
            x_binned[i] = method(@view(x[p[bins[i]+1:bins[i+1]-1]]))
            y_binned[i] = method(@view(y[p[bins[i]+1:bins[i+1]-1]]))
        end
        
        return x_binned, y_binned, bin_count 
    end



    """
        epsg2epsg(x, y, from_epsg, to_epsg)

    Returns `x`, `y` in `to_epsg` projection
    """
    function epsg2epsg(
        x,
        y,
        from_epsg::String,
        to_epsg::String;
        parse_output=true
    )
        # this function was slower when using @threads, tested with length(x) == 10000

        # build transformation 
        trans = Proj.Transformation(from_epsg, to_epsg, always_xy=true)

        # project points

        data = trans.(x, y)


        if parse_output
            if x isa Vector{}
                x = getindex.(data, 1)
                y = getindex.(data, 2)
            else
                x = getindex(data, 1)
                y = getindex(data, 2)
            end
            return x, y
        else
            return data
        end
    end

end
