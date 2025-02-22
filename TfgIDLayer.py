# coding=utf-8

from tf_geometric.nn.conv.gcn import gcn_mapper, gcn_norm_edge, gcn_build_cache_for_graph
from tf_geometric.nn import mean_reducer, max_reducer, sum_reducer
import tensorflow as tf
import tf_sparse as tfs
import warnings
from tf_geometric.utils.graph_utils import add_self_loop_edge
from sparse_adj import SparseAdj
from sparse_ops import sparse_diag_matmul, diag_sparse_matmul

from tf_geometric.nn.kernel.segment import segment_softmax


class IDSAGE(tf.keras.Model):
    """
    GraphSAGE: `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """

    def __init__(self, units, activation=tf.nn.relu, use_bias=True, concat=True,
                 normalize=False,
                 kernel_regularizer=None, id_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """
        :param units:
        :param activation:
        :param use_bias:
        :param concat:
        :param normalize:
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat
        self.normalize = normalize

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.kernel_regularizer = kernel_regularizer
        self.id_regularizer = id_regularizer  # This isn't actually used, but in case we want to add one later.
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.id_kernel = None
        self.neighbor_kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        # Add in parameters for the id portion of the network
        self.id_kernel = self.add_weight("id_kernel", shape=[num_features, kernel_units], initializer="glorot_uniform",
                                         regularizer=self.id_regularizer)
        self.neighbor_kernel = self.add_weight("neighbor_kernel", shape=[num_features, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """
        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        if len(inputs) == 4:
            x, edge_index, id_index, edge_weight = inputs
        else:
            x, edge_index, id_index = inputs
            edge_weight = None

        # Derived from tfg.nn.conv.mean_graph_sage([...])
        # -----------------------------------------------
        num_nodes = tf.shape(x)[0]
        # num_edges = tf.shape(edge_index)[1]

        row, col = edge_index[0], edge_index[1]
        repeated_x = tf.gather(x, row)
        neighbor_x = tf.gather(x, col)

        if edge_weight is not None:
            neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

        neighbor_reduced_msg = mean_reducer(neighbor_x, row, num_nodes=num_nodes)

        neighbor_msg = neighbor_reduced_msg @ self.neighbor_kernel
        h = x @ self.self_kernel

        x_id = tf.gather(x, id_index, axis=0)  # Gather the appropriate inputs for the given id_index values
        h_id = x_id @ self.id_kernel  # Perform a matrix multiplication using the id_kernel

        id_index = tf.reshape(id_index, [-1, 1])
        h = tf.tensor_scatter_nd_add(h, id_index, h_id)  # Recombine the hidden id values with the other features

        if self.concat:
            h = tf.concat([h, neighbor_msg], axis=1)
        else:
            h = h + neighbor_msg

        if self.bias is not None:
            h += self.bias
        if self.activation is not None:
            h = self.activation(h)
        if self.normalize:
            h = tf.nn.l2_normalize(h, axis=-1)
        return h


class IDGIN(tf.keras.Model):
    """
    Graph Isomorphism Network  Layer
    """

    def __init__(self, mlp_model, mlpid_model, eps=0, train_eps=False, *args, **kwargs):
        """
        :param mlp_model: A neural network (multi-layer perceptrons).
        :param eps: float, optional, (default: :obj:`0.`).
        :param train_eps: Boolean, Whether the eps is trained.
        :param activation: Activation function to use.
        """
        super().__init__(*args, **kwargs)
        self.mlp_model = mlp_model
        self.mlp_id = mlpid_model  # Include a second MLP for the id kernel.

        self.eps = eps
        if train_eps:
            self.eps = self.add_weight("eps", shape=[], initializer="zeros")

    def call(self, inputs, cache=None, training=None, mask=None):
        """
        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GIN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 4:  # Process the inputs appropriately based on the complexity of the input
            x, edge_index, id_index, _ = inputs
        else:
            x, edge_index, id_index = inputs

        # TODO: maybe remove_self_loops to get edge_index ref line 370 of 'idconv.py'
        num_nodes = tf.shape(x)[0]
        sparse_adj = SparseAdj(edge_index, shape=[num_nodes, num_nodes])
        neighbor_h = sparse_adj @ x
        h = x * (1.0 + self.eps) + neighbor_h

        h_id = tf.gather(h, id_index, axis=0)  # retrieve the hidden values for the given id_index
        h_id = self.mlp_id(h_id, training=training)  # Process the id with the MLP for the id kernel
        h = self.mlp_model(h, training=training)
        id_index = tf.reshape(id_index, [-1, 1])
        h = tf.tensor_scatter_nd_add(h, id_index, h_id)  # Recombine the hidden state for the id with the rest of the
        # hidden state
        return h


class IDGAT(tf.keras.Model):

    def __init__(self, units,
                 attention_units=None,
                 activation=None,
                 use_bias=True,
                 num_heads=1,
                 split_value_heads=True,
                 query_activation=tf.nn.relu,
                 key_activation=tf.nn.relu,
                 drop_rate=0.0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        """
        :param units: Positive integer, dimensionality of the output space.
        :param attention_units: Positive integer, dimensionality of the output space for Q and K in attention.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param num_heads: Number of attention heads.
        :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
            Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
        :param query_activation: Activation function for Q in attention.
        :param key_activation: Activation function for K in attention.
        :param drop_rate: Dropout rate.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_units = units if attention_units is None else attention_units
        self.drop_rate = drop_rate

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.kernel = None
        self.kernel_id = None
        self.bias = None

        self.activation = activation
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.split_value_heads = split_value_heads

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.query_bias = self.add_weight("query_bias", shape=[self.attention_units],
                                          initializer="zeros", regularizer=self.bias_regularizer)

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, self.attention_units],
                                          initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.key_bias = self.add_weight("key_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        # Include a kernel for the id portion of the network.
        self.kernel_id = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: List of graph info: [x, edge_index] or [x, edge_index, edge_weight].
            Note that the edge_weight will not be used.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 4:  # Process the inputs appropriately based on the complexity of the input
            x, edge_index, id_index, edge_weight = inputs
        else:
            x, edge_index, id_index = inputs
            edge_weight = None
        # Pass all required data to the gat_id function for further processing
        return gat_id(x, edge_index, id_index,
                      self.query_kernel, self.query_bias, self.query_activation,
                      self.key_kernel, self.key_bias, self.key_activation,
                      self.kernel, self.kernel_id,self.bias, self.activation,
                      num_heads=self.num_heads,
                      split_value_heads=self.split_value_heads,
                      drop_rate=self.drop_rate,
                      training=training)


def gat_id(x, edge_index, id_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, kernel_id,bias=None, activation=None, num_heads=1,
        split_value_heads=True, drop_rate=0.0, training=False):
    """
    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param query_kernel: Tensor, shape: [num_features, num_query_features], weight for Q in attention
    :param query_bias: Tensor, shape: [num_query_features], bias for Q in attention
    :param query_activation: Activation function for Q in attention.
    :param key_kernel: Tensor, shape: [num_features, num_key_features], weight for K in attention
    :param key_bias: Tensor, shape: [num_key_features], bias for K in attention
    :param key_activation: Activation function for K in attention.
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param num_heads: Number of attention heads.
    :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
        Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
    :param drop_rate: Dropout rate.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tf.shape(x)[0]

    # self-attention
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]

    x_is_sparse = isinstance(x, tf.sparse.SparseTensor)

    if x_is_sparse:
        Q = tf.sparse.sparse_dense_matmul(x, query_kernel)
    else:
        Q = x @ query_kernel
    Q += query_bias
    if query_activation is not None:
        Q = query_activation(Q)
    Q = tf.gather(Q, row)

    if x_is_sparse:
        K = tf.sparse.sparse_dense_matmul(x, key_kernel)
    else:
        K = x @ key_kernel
    K += key_bias
    if key_activation is not None:
        K = key_activation(K)
    K = tf.gather(K, col)

    if x_is_sparse:
        V = tf.sparse.sparse_dense_matmul(x, kernel)
    else:
        x_id = tf.gather(x, id_index, axis=0)  # retrieve the values for the given id_indices
        h_id = x_id @ kernel_id  # Perform a matrix multiplication against the kernel for the id portion of the network
    
        h = x @ kernel
        id_index = tf.reshape(id_index, [-1, 1])  # Recombine the id portion with the remainder of the network values
        V = tf.tensor_scatter_nd_add(h, id_index, h_id)

    # xxxxx_ denotes the multi-head style stuff
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    # splited queries and keys are modeled as virtual vertices
    qk_edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)

    scale = tf.math.sqrt(tf.cast(tf.shape(Q_)[-1], tf.float32))
    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1) / scale

    # new implementation based on SparseAdj
    num_nodes_ = num_nodes * num_heads
    sparse_att_adj = SparseAdj(qk_edge_index_, att_score_, [num_nodes_, num_nodes_]) \
        .softmax(axis=-1) \
        .dropout(drop_rate, training=training)
    

    if split_value_heads:
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    else:
        V_ = V
        edge_index_ = tf.tile(edge_index, [1, num_heads])
        sparse_att_adj = SparseAdj(edge_index_, sparse_att_adj.edge_weight, [num_nodes, num_nodes])
#
    h_ = sparse_att_adj @ V_
    
   
    #normed_att_score_ = segment_softmax(att_score_, qk_edge_index_[0], num_nodes * num_heads)
    #if training and drop_rate > 0.0:
    #    normed_att_score_ = tf.compat.v2.nn.dropout(normed_att_score_, drop_rate)
    #if split_value_heads:
    #    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    #    edge_index_ = qk_edge_index_
    #else:
    #    V_ = V
    #    edge_index_ = tf.tile(edge_index, [1, num_heads])
    #h_ = aggregate_neighbors(
    #    V_, edge_index_, normed_att_score_,
    #    gcn_mapper,
    #    sum_reducer,
    #    identity_updater
    #)    
    



    if split_value_heads:
        h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)
    else:
        h = h_ / num_heads

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h


class IDGCN(tf.keras.Model):
    """
    Graph Convolutional Layer
    """

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        # Include a kernel for the id portion of the network.
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

        if len(inputs) == 4:  # Process the inputs appropriately based on the complexity of the input
            x, edge_index, id_index, edge_weight = inputs
        else:
            x, edge_index, id_index = inputs
            edge_weight = None
        # Pass all required data to the gat_id function for further processing
        return gcn_id(x, edge_index, id_index, edge_weight, self.kernel, self.kernel_id, self.bias,
                      activation=self.activation, renorm=self.renorm, improved=self.improved, cache=cache)


def gcn_id(x, edge_index, id_index, edge_weight, kernel, kernel_id, bias=None, activation=None,
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
        
        x_id = tf.gather(x, id_index, axis=0)  # gather the parameters related to the id portion of the network
        h_id = x_id @ kernel_id  # Perform a matrix multiplication with the id kernel
        h = x @ kernel
        # h.index_add_(0, id_index, h_id)
        id_index = tf.reshape(id_index, [-1, 1])  # recombine the id values with the rest of the values
        h = tf.tensor_scatter_nd_add(h, id_index, h_id)

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
