module GraphSAGE
    using Statistics;
    using StatsBase: sample;
    using LightGraphs;
    using Flux;

    export graph_encoder;

    struct AGG{F}
        S::String;
        L::F;
    end

    function AGG(S::String, dim_h::Int, dim_e::Int, σ=relu)
        """"
        dim_h: dimension of vertice embedding
        dim_e: dimension of edge embedding
        """

        @assert S in ["Mean", "Max", "Sum", "MeanPooling", "MaxPooling", "SumPooling"];

        if S in ["Mean", "Max", "Sum"]
            return AGG(S, nothing);
        else
            return AGG(S, Dense(dim_h+dim_e, dim_h+dim_e, σ));
        end
    end

    function (c::AGG)(he::Vector)
        S, L = c.S, c.L;

        if S == "Mean"
            return mean(he);
        elseif S == "Max"
            return max.(he...);
        elseif S == "Sum"
            return sum(he);
        elseif S == "MeanPooling"
            return mean(L.(he));
        elseif S == "MaxPooling"
            return max.(L.(he)...);
        elseif S == "SumPooling"
            return sum(L.(he));
        end
    end

    Flux.@treelike AGG;


    # sampler & aggregator
    struct SAGE{F}
        T::F;
        k::Int;
        A::AGG;
        # default value (when vertex has no edge)
        z::AbstractVector;
    end

    function SAGE(T::F, k::Int, S::String, dim_h::Int, dim_e::Int, σ=relu) where {F}
        return SAGE(T, k, AGG(S, dim_h, dim_e, σ), zeros(dim_h+dim_e));
    end

    function (c::SAGE)(G::AbstractGraph, node_list::Vector{Int}, node_features::Function, edge_features::Function)
        T, k, A, z = c.T, c.k, c.A, c.z;

        sampled_nbrs_list = Vector{Vector{Int}}();
        for u in node_list
            nbrs = inneighbors(G, u);
            push!(sampled_nbrs_list, length(nbrs) > k ? sample(nbrs, k, replace=false) : nbrs);
        end

        # compute hidden vector of unique neighbors
        unique_nodes = union(node_list, sampled_nbrs_list...);
        u2i = Dict{Int,Int}(u=>i for (i,u) in enumerate(unique_nodes));

        # if this SAGE is not a leaf, then call the child Transformer to get node representation at previous layer
        if T != nothing
            h0 = T(G, unique_nodes, node_features, edge_features);
        else
            h0 = [convert(Vector{Float32}, node_features(u)) for u in unique_nodes];
        end

        # each vector can be decomposed as [h(v)*, edge_features(v,u)*, h(u)], where * means 'aggregated across v'
        heh = Vector{AbstractVector}();
        for (u, sampled_nbrs) in zip(node_list, sampled_nbrs_list)
            he = length(sampled_nbrs) != 0 ? A([vcat(h0[u2i[v]], convert(Vector{Float32}, edge_features(v,u))) for v in sampled_nbrs]) : z;
            push!(heh, vcat(he, h0[u2i[u]]));
        end

        return heh;
    end

    Flux.@treelike SAGE;



    # transformer
    struct Transformer{F}
        S::SAGE;
        L::F;
    end

    function Transformer(S::SAGE, dim_h0::Int, dim_h1::Int, dim_e::Int, σ=relu)
        L = Dense(dim_h0*2+dim_e, dim_h1, σ);

        return Transformer(S, L);
    end

    function (c::Transformer)(G::AbstractGraph, node_list::Vector{Int}, node_features::Function, edge_features::Function)
        S, L = c.S, c.L;

        h1 = L.(S(G, node_list, node_features, edge_features));

        return h1;
    end

    Flux.@treelike Transformer;



    # graph encoder
    function graph_encoder(dim_in::Int, dim_out::Int, dim_h::Int, dim_e::Int, layers::Vector{String};
                           ks::Vector{Int}=repeat([typemax(Int)], length(layers)), σ=relu)
        @assert length(layers) > 0;
        @assert length(layers) == length(ks);

        sage = SAGE(nothing, ks[1], layers[1], dim_in, dim_e, σ);
        if length(layers) == 1
            # single layer, directly output
            tsfm = Transformer(sage, dim_in, dim_out, dim_e, σ);
        else
            # multiple layer, first encode to hidden
            tsfm = Transformer(sage, dim_in, dim_h, dim_e, σ);

            # the inner layers, hidden to hidden
            for i in 2:length(layers)-1
                sage = SAGE(tsfm, ks[i], layers[i], dim_h, dim_e, σ);
                tsfm = Transformer(sage, dim_h, dim_h, dim_e, σ);
            end

            sage = SAGE(tsfm, ks[end], layers[end], dim_h, dim_e, σ);
            tsfm = Transformer(sage, dim_h, dim_out, dim_e, σ);
        end

        return tsfm;
    end
end
