# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GNN model from Flax ogbg_molpcba example, see:

https://github.com/google/flax/tree/main/examples/ogbg_molpcba
"""

from collections.abc import Callable, Sequence

from flax import linen as nn

import jax
import jax.numpy as jnp
import jraph


def add_graphs_tuples(graphs: jraph.GraphsTuple,
                      other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Adds the nodes, edges and global features from other_graphs to graphs."""
  return graphs._replace(
      nodes=graphs.nodes + other_graphs.nodes,
      edges=graphs.edges + other_graphs.edges,
      globals=graphs.globals + other_graphs.globals)


class MLP(nn.Module):
  """A multi-layer perceptron."""

  feature_sizes: Sequence[int]
  dropout_rate: float = 0
  deterministic: bool = True
  activation: Callable[[jax.Array], jax.Array] = nn.relu

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for size in self.feature_sizes:
      x = nn.Dense(features=size)(x)
      x = self.activation(x)
      x = nn.Dropout(
          rate=self.dropout_rate, deterministic=self.deterministic)(x)
    return x


class GraphNet(nn.Module):
  """A complete Graph Network model defined with Jraph."""

  latent_size: int
  num_mlp_layers: int
  message_passing_steps: int
  output_globals_size: int
  dropout_rate: float = 0
  skip_connections: bool = True
  use_edge_model: bool = True
  layer_norm: bool = True
  deterministic: bool = True

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # We will first linearly project the original features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.latent_size),
        embed_edge_fn=nn.Dense(self.latent_size),
        embed_global_fn=nn.Dense(self.latent_size))
    processed_graphs = embedder(graphs)

    # Now, we will apply a Graph Network once for each message-passing round.
    mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
    for _ in range(self.message_passing_steps):
      if self.use_edge_model:
        update_edge_fn = jraph.concatenated_args(
            MLP(mlp_feature_sizes,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic))
      else:
        update_edge_fn = None

      update_node_fn = jraph.concatenated_args(
          MLP(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))
      update_global_fn = jraph.concatenated_args(
          MLP(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))

      graph_net = jraph.GraphNetwork(
          update_node_fn=update_node_fn,
          update_edge_fn=update_edge_fn,
          update_global_fn=update_global_fn)

      if self.skip_connections:
        processed_graphs = add_graphs_tuples(
            graph_net(processed_graphs), processed_graphs)
      else:
        processed_graphs = graph_net(processed_graphs)

      if self.layer_norm:
        processed_graphs = processed_graphs._replace(
            nodes=nn.LayerNorm()(processed_graphs.nodes),
            edges=nn.LayerNorm()(processed_graphs.edges),
            globals=nn.LayerNorm()(processed_graphs.globals),
        )

    # Since our graph-level predictions will be at globals, we will
    # decode to get the required output logits.
    decoder = jraph.GraphMapFeatures(
        embed_global_fn=nn.Dense(self.output_globals_size))
    processed_graphs = decoder(processed_graphs)

    return processed_graphs


class GraphConvNet(nn.Module):
  """A Graph Convolution Network + Pooling model defined with Jraph."""

  latent_size: int
  num_mlp_layers: int
  message_passing_steps: int
  output_globals_size: int
  dropout_rate: float = 0
  skip_connections: bool = True
  layer_norm: bool = True
  deterministic: bool = True
  pooling_fn: Callable[[jax.Array, jax.Array, jax.Array],
                       jax.Array] = jraph.segment_mean

  def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pooling operation, taken from Jraph."""

    # Equivalent to jnp.sum(n_node), but JIT-able.
    sum_n_node = graphs.nodes.shape[0]
    # To aggregate nodes from each graph to global features,
    # we first construct tensors that map the node to the corresponding graph.
    # Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
    n_graph = graphs.n_node.shape[0]
    node_graph_indices = jnp.repeat(
        jnp.arange(n_graph),
        graphs.n_node,
        axis=0,
        total_repeat_length=sum_n_node)
    # We use the aggregation function to pool the nodes per graph.
    pooled = self.pooling_fn(graphs.nodes, node_graph_indices, n_graph)
    return graphs._replace(globals=pooled)

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # We will first linearly project the original node features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.latent_size))
    processed_graphs = embedder(graphs)

    # Now, we will apply the GCN once for each message-passing round.
    for _ in range(self.message_passing_steps):
      mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
      update_node_fn = jraph.concatenated_args(
          MLP(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))
      graph_conv = jraph.GraphConvolution(
          update_node_fn=update_node_fn, add_self_edges=True)

      if self.skip_connections:
        processed_graphs = add_graphs_tuples(
            graph_conv(processed_graphs), processed_graphs)
      else:
        processed_graphs = graph_conv(processed_graphs)

      if self.layer_norm:
        processed_graphs = processed_graphs._replace(
            nodes=nn.LayerNorm()(processed_graphs.nodes),
        )

    # We apply the pooling operation to get a 'global' embedding.
    processed_graphs = self.pool(processed_graphs)

    # Now, we decode this to get the required output logits.
    decoder = jraph.GraphMapFeatures(
        embed_global_fn=nn.Dense(self.output_globals_size))
    processed_graphs = decoder(processed_graphs)

    return processed_graphs
