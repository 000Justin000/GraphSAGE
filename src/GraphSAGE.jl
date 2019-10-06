module GraphSAGE
    using StatsBase: sample, Weights;
    using LightGraphs;
    using SimpleWeightedGraphs;
    using Flux;

    abstract type SAGE end;

    # mean aggregator
    struct MeanSAGE{F} <: SAGE
        f::F;
        k::Int64;
        G::AbstractGraph;
    end

    function (c::MeanSAGE)(node_list)
        f, k, G = c.f, c.k, c.G;

        # perform weighted sample
        # note we are sampling without replacement here, hense the expected mean equals the true weighted average
        sampled_nbrs_list = Vector{Vector{Int64}}();
        for u in node_list
            nbrs = inneighbors(G, u);
            upbs = Vector{Float64}([weights(G)[u, v] for v in nbrs]);
            push!(sampled_nbrs_list, length(nbrs) != 0 ? sample(nbrs, Weights(upbs), k) : []);
        end

        # compute hidden vector of unique neighbors
        unique_nodes = union(node_list, sampled_nbrs_list...);
        u2i = Dict{Int64,Int64}(u=>i for (i,u) in enumerate(unique_nodes));
        hh0 = f(unique_nodes);

        @assert length(hh0) > 0 "non of the vertices has incoming edge"
        sz = size(hh0[1]);

        # compute the mean hidden vector of the sampled neighbors
        hh1_ = Vector{AbstractVector}();
        for (u, sampled_nbrs) in zip(node_list, sampled_nbrs_list)
            h_nbrs = length(sampled_nbrs) != 0 ? sum(hh0[u2i[v]] for v in sampled_nbrs) / length(sampled_nbrs) : zeros(sz);
            push!(hh1_, vcat(hh0[u2i[u]], h_nbrs));
        end

        return hh1_;
    end

    Flux.@treelike MeanSAGE;


    # transformer
    struct Transformer{F <:SAGE,T}
        f::F;
        L::T;
    end

    function Transformer(f::SAGE, dim_h0::Integer, dim_h1::Integer, σ=relu)
        L = Dense(dim_h0*2, dim_h1, σ);

        return Transformer(f, L);
    end

    function (c::Transformer)(node_list)
        println(node_list);
        f, L = c.f, c.L;

        hh1 = L.(f(node_list));

        return hh1;
    end

    Flux.@treelike Transformer;
end
