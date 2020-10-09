module GraphSAGE
    using Statistics;
    using StatsBase: sample;
    using LightGraphs;
    using Flux;

    export graph_encoder;

    # sampler & aggregator
    struct SAGE{Q,F}
        P::Q;              # pointer to the previous layer
        k::Int;            # max number of neighbors to sample
        z::AbstractVector; # default value (when vertex has no edge)
        L::F;              # neighborhood vector transformation functor
        A::Function;       # aggregation function
        C::Function;       # combination function
    end

    function SAGE(P::Q, k::Int, dim_h::Int; pooling::Bool=false, σ::Function=relu, agg_type::String="Mean", cmb_type::String="Mean") where {Q}
        z = zeros(Float32, dim_h);
        L = pooling ? Dense(dim_h, dim_h, σ) : identity;
        
        if agg_type == "Mean"
            A = HN -> Flux.mean(HN, dims=2)[:];
        elseif agg_type == "Sum"
            A = HN -> Flux.sum(HN, dims=2)[:];
        elseif agg_type == "Max"
            A = HN -> Flux.maximum(HN, dims=2)[:];
        else
            error("unexpected aggregation type");
        end

        if cmb_type == "AVG"
            C = (hu, hn, d) -> (hu + hn*d) / (d+1);
        elseif cmb_type == "CAT"
            C = (hu, hn, d) -> vcat(hu, hn);
        else
            error("unexpected combination type");
        end

        return SAGE(P, k, z, L, A, C);
    end

    function (sage::SAGE)(G::AbstractGraph, node_list::Vector{Int}, node_to_features::Function)
        P, k, z, L, A, C = sage.P, sage.k, sage.z, sage.L, sage.A, sage.C;

        sampled_nbrs_list = Vector{Vector{Int}}();
        for u in node_list
            nbrs = inneighbors(G, u);
            push!(sampled_nbrs_list, length(nbrs) > k ? sample(nbrs, k, replace=false) : nbrs);
        end

        # compute hidden vector of unique neighbors
        unique_nodes = union(node_list, sampled_nbrs_list...);
        u2i = Dict{Int,Int}(u=>i for (i,u) in enumerate(unique_nodes));

        # if this SAGE is not a leaf, then call the child Transformer to get node representation at previous layer
        if P !== nothing
            h0 = P(G, unique_nodes, node_to_features);
        else
            h0 = [f32(node_to_features(u)) for u in unique_nodes];
        end

        # computed new hidden vectors
        h1 = Vector{AbstractVector}();
        for (u, sampled_nbrs) in zip(node_list, sampled_nbrs_list)
            d = length(sampled_nbrs);

            # hidden state of the current vertex
            hu = h0[u2i[u]];
            # aggregate the hidden states of the neighboring vertices
            hn = (d == 0) ? z : A(L(Flux.hcat([h0[u2i[v]] for v in sampled_nbrs]...)));

            # either take the weighted average or concatenate the two
            hh = C(hu, hn, d);

            push!(h1, hh);
        end

        return h1;
    end

    Flux.@treelike SAGE;



    # transformer
    struct TSFM{Q,F}
        P::Q;              # pointer to the previous layer
        L::F;              # transformation functor applied to the hidden vector of each vertex
    end

    function (tsfm::TSFM)(G::AbstractGraph, node_list::Vector{Int}, node_to_features::Function)
        P, L = tsfm.P, tsfm.L;

        if P !== nothing
            h0 = P(G, node_list, node_to_features);
        else
            h0 = [f32(node_to_features(u)) for u in node_list];
        end

        return L.(h0);
    end

    Flux.@treelike TSFM;



    # graph encoder
    function graph_encoder(method::String, dim_in::Int, dim_out::Int, dim_h::Int, nl::Int; k::Int=typemax(Int), σ=relu)
    """
    Args:
       method: [SAGE_GCN, SAGE_Mean, SAGE_Max, SAGE_MaxPooling, SAGE_SmoothCLS]
       dim_in: node feature dimension
      dim_out: embedding dimension
        dim_h: hidden dimension
            k: max number of sampled neighbors to pull

    Returns:
      encoder: a model that takes 1) graph topology 2) vertex features 3) vertices to be encoded 
               as inputs and gives vertex embeddings / predictive probabilities as output
    """
        if method == "SAGE_GCN"
            agg_type = "Mean";
            cmb_type = "AVG";
            pooling = false;
        elseif method == "SAGE_Mean"
            agg_type = "Mean";
            cmb_type = "CAT";
            pooling = false;
        elseif method == "SAGE_Max"
            agg_type = "Max";
            cmb_type = "CAT";
            pooling = false;
        elseif method == "SAGE_MaxPooling"
            agg_type = "Max";
            cmb_type = "CAT";
            pooling = true;
        elseif method == "SAGE_SmoothCLS"
            agg_type = "Mean";
            cmb_type = "AVG";
            pooling = false;
        else
            error("unexpected method");
        end

        if method in ["SAGE_GCN", "SAGE_Mean", "SAGE_Max", "SAGE_MaxPooling"]
            sage = SAGE(nothing, k, dim_in; pooling=pooling, σ=σ, agg_type=agg_type, cmb_type=cmb_type);
            nc = (cmb_type == "CAT") ? 2 : 1;

            if nl == 1
                # single layer, directly output
                tsfm = TSFM(sage, Dense(dim_in*nc, dim_out, σ));
            else
                # multiple layer, first encode to hidden
                tsfm = TSFM(sage, Dense(dim_in*nc, dim_h, σ));

                # the inner layers, hidden to hidden
                for i in 2:nl-1
                    sage = SAGE(tsfm, k, dim_h; pooling=pooling, σ=σ, agg_type=agg_type, cmb_type=cmb_type);
                    tsfm = TSFM(sage, Dense(dim_h*nc, dim_h, σ));
                end

                sage = SAGE(tsfm, k, dim_h; pooling=pooling, σ=σ, agg_type=agg_type, cmb_type=cmb_type);
                tsfm = TSFM(sage, Dense(dim_h*nc, dim_h, σ));
            end

            encoder = tsfm;
        elseif method == "SAGE_SmoothCLS"
            mlp = Chain(Dense(dim_in, dim_h, σ), Dense(dim_h, dim_h, σ), Dense(dim_h, dim_out, softmax));
            tsfm = TSFM(nothing, mlp);

            sage = SAGE(tsfm, k, dim_out; pooling=pooling, σ=σ, agg_type=agg_type, cmb_type=cmb_type);
            for i in 2:nl-1
                sage = SAGE(sage, k, dim_out; pooling=pooling, σ=σ, agg_type=agg_type, cmb_type=cmb_type);
            end

            encoder = sage;
        end

        return encoder;
    end
end
