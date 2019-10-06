module GraphSAGE
    using StatsBase: sample, Weights;
    using LightGraphs;
    using SimpleWeightedGraphs;
    using Flux;

    export graph_encoder;

    abstract type SAGE end;

    # mean aggregator
    struct MeanSAGE{F} <: SAGE
        f::F;
        k::Int;
    end

    function (c::MeanSAGE)(G::AbstractGraph, node_list::Vector{Int})
        f, k = c.f, c.k;

        # perform weighted sample
        # note we are sampling without replacement here, hense the expected mean equals the true weighted average
        sampled_nbrs_list = Vector{Vector{Int}}();
        for u in node_list
            nbrs = inneighbors(G, u);
            upbs = Vector{Float64}([weights(G)[u, v] for v in nbrs]);
            push!(sampled_nbrs_list, length(nbrs) != 0 ? sample(nbrs, Weights(upbs), k) : []);
        end

        # compute hidden vector of unique neighbors
        unique_nodes = union(node_list, sampled_nbrs_list...);
        u2i = Dict{Int,Int}(u=>i for (i,u) in enumerate(unique_nodes));
        hh0 = f(G, unique_nodes);

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

    function (c::Transformer)(G::AbstractGraph, node_list::Vector{Int})
        f, L = c.f, c.L;

        hh1 = L.(f(G, node_list));

        return hh1;
    end

    Flux.@treelike Transformer;



    # graph encoder
    function graph_encoder(features::Vector, dim_in::Integer, dim_out::Integer, dim_h::Integer, layers::Vector{String}, ks::Vector{Int}, σ=relu)
        @assert length(layers) > 0 "number of layers must be positive"

        # first aggregator always pull input features
        agg = (@eval $Symbol(layers[1]))((G,node_list) -> features[node_list], ks[1]);
        if length(layers) == 1
            # single layer, directly output
            tsf = Transformer(agg, dim_in, dim_out, σ);
        else
            # multiple layer, first encode to hidden
            tsf = Transformer(agg, dim_in, dim_h, σ);

            # the inner layers, hidden to hidden
            for i in 2:length(layers)-1
                agg = (@eval $Symbol(layers[i]))(tsf, ks[i]);
                tsf = Transformer(agg, dim_h, dim_h, σ);
            end

            agg = (@eval $Symbol(layers[end]))(tsf, ks[end]);
            tsf = Transformer(agg, dim_h, dim_out, σ);
        end

        return tsf;
    end
end
