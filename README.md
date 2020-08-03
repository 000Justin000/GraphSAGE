## Julia GraphSAGE Implementation
#### Author: Junteng Jia

Basic reference Julia implementation of [GraphSAGE](https://github.com/williamleif/GraphSAGE).
The goal of this project is to provide a native and efficient implementation for the Julia graph learning community.
The code is intended to be simpler, more extensible, and easier to work with than the original TensorFlow version.

Currently, we support the supervised versions of GraphSAGE-GCN, GraphSAGE-mean, GraphSAGE-max, and GraphSAGE-pooling.

#### Requirements

Julia > 1.0, and Flux = 0.9 is required.

#### Installation

```julia
] add https://github.com/000Justin000/GraphSAGE
```

#### Basic Usage

First, define a vertex encoder with the following function call:
```julia
encoder = graph_encoder(dim_in::Int, dim_out::Int, dim_h::Int, layers::Vector{String}; ks::Vector{Int}=repeat([typemax(Int)], length(layers)), σ=relu)
"""
Args:
  dim_in: node feature dimension
  dim_out: embedding dimension
  dim_h: hidden dimension
  layers: each is a convolution layer of a certain convolution type
  ks: max number of sampled neighbors to pull
"""
```

Once you define the graph encoder, the way to use it is:
```julia
embeddings = encoder(G::AbstractGraph, node_list::Vector{Int}, node_features::Function)
"""
Args:
  G: is a lightGraph object
  node_list: the set of vertices you want to compute embedding
  node_features: a function that maps a list of vertex indices to their features
"""
```

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
