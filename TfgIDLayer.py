# coding=utf-8

from tf_geometric.nn.conv.gcn import  gcn_build_cache_for_graph
import tensorflow as tf
import warnings
from tf_geometric.utils.graph_utils import add_self_loop_edge
from tf_geometric.sparse.sparse_adj import SparseAdj
from tf_geometric.sparse.sparse_ops import sparse_diag_matmul, diag_sparse_matmul





class IDGCN(tf.keras.Model):
    """
    Graph Convolutional Layer
    """

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.kernel_id = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def __init__(self, units, activation=None,
                 use_bias=True,
                 renorm=True, improved=False,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """
        :param units: Positive integer, dimensionality of the output space.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units

        self.activation = activation
        self.use_bias = use_bias

        self.kernel = None
        self.kernel_id = None
        self.bias = None

        self.renorm = renorm
        self.improved = improved

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_build_cache_for_graph(graph, self.renorm, self.improved, override=override)

    def cache_normed_edge(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        .. deprecated:: 0.0.56
            Use ``build_cache_for_graph`` instead.
        """
        warnings.warn("'GCN.cache_normed_edge(graph, override)' is deprecated, use 'GCN.build_cache_for_graph(graph, override)' instead", DeprecationWarning)
        return self.build_cache_for_graph(graph, override=override)

    def call(self, inputs, cache=None, training=None, mask=None):
        """
        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return gcn_id(x, edge_index, edge_weight, self.kernel, self.kernel_id, self.bias,
                   activation=self.activation, renorm=self.renorm, improved=self.improved, cache=cache)



def gcn_id(x, edge_index, edge_weight, kernel, bias=None, activation=None,
        renorm=True, improved=False, cache=None):
    """
    Functional API for Graph Convolutional Networks.
    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        To use @tf_utils.function with gcn, you should cache the noremd edge information before the first call of the gcn.
        - (1) If you're using OOP APIs tfg.layers.GCN:
              gcn_layer.build_cache_for_graph(graph)
        - (2) If you're using functional API tfg.nn.gcn:
              from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph
              gcn_build_cache_for_graph(graph)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tf.shape(x)[0]

    sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, renorm, improved, cache)

    # SparseTensor is usually used for one-hot node features (For example, feature-less nodes.)
    if isinstance(x, tf.sparse.SparseTensor):
        h = tf.sparse.sparse_dense_matmul(x, kernel)
    else:
        h = x @ kernel

    h = normed_sparse_adj @ h

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h

def gcn_norm_adj(sparse_adj, renorm=True, improved=False, cache: dict = None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.
    :param sparse_adj: SparseAdj, sparse adjacency matrix.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    if cache is not None:
        cache_key = compute_cache_key(renorm, improved)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            return cached_data

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        sparse_adj = sparse_adj.add_self_loop(fill_weight=fill_weight)

    deg = sparse_adj.reduce_sum(axis=-1)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    # (D^(-1/2)A)D^(-1/2)
    normed_sparse_adj = sparse_diag_matmul(diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

    if not renorm:
        normed_sparse_adj = normed_sparse_adj.add_self_loop(fill_weight=fill_weight)

    if cache is not None:
        cache[cache_key] = normed_sparse_adj

    return normed_sparse_adj



