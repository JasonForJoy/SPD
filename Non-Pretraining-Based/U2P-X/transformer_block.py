import tensorflow as tf

# main function
def block(
    Q, K, V, 
    Q_lengths, K_lengths, 
    attention_type='dot', 
    is_layer_norm=True, 
    is_mask=True, mask_value=-2**32+1,
    drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]
    '''
    att = attention(Q, K, V, 
                    Q_lengths, K_lengths, 
                    attention_type=attention_type, 
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm'):
            y = layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y)
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm'):
            w = layer_norm_debug(y + z)
    else:
        w = y + z
    return w

def attention(
    Q, K, V, 
    Q_lengths, K_lengths, 
    attention_type='dot', 
    is_mask=True, mask_value=-2**32+1,
    drop_prob=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = dot_sim(Q, K) #[batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = bilinear_sim(Q, K)

    if is_mask:
        _mask = mask(Q_lengths, K_lengths, Q_time, K_time) #[batch, Q_time, K_time]
        logits = _mask * logits + (1 - _mask) * mask_value
    
    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return weighted_sum(attention, V)

def dot_sim(x, y, is_nor=True):
    '''calculate dot similarity with two tensor.

    Args:
        x: a tensor with shape [batch, time_x, dimension]
        y: a tensor with shape [batch, time_y, dimension]
    
    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        AssertionError: if
            the shapes of x and y are not match.
    '''
    assert x.shape[-1] == y.shape[-1]

    sim = tf.einsum('bik,bjk->bij', x, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return result

def bilinear_sim(x, y, is_nor=True):
    '''calculate bilinear similarity with two tensor.
    Args:
        x: a tensor with shape [batch, time_x, dimension_x]
        y: a tensor with shape [batch, time_y, dimension_y]
    
    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        ValueError: if
            the shapes of x and y are not match;
            bilinear matrix reuse error.
    '''
    M = tf.get_variable(
        name="bilinear_matrix", 
        shape=[x.shape[-1], y.shape[-1]],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    sim = tf.einsum('bik,kl,bjl->bij', x, M, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1] * y.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim

def mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''Return a mask tensor representing the first N positions of each row and each column.

    Args:
        row_lengths: a tensor with shape [batch]
        col_lengths: a tensor with shape [batch]

    Returns:
        a mask tensor with shape [batch, max_row_length, max_col_length]

    Raises:
    '''
    row_mask = tf.sequence_mask(row_lengths, max_row_length) #bool, [batch, max_row_len]
    col_mask = tf.sequence_mask(col_lengths, max_col_length) #bool, [batch, max_col_len]

    row_mask = tf.cast(tf.expand_dims(row_mask, -1), tf.float32)
    col_mask = tf.cast(tf.expand_dims(col_mask, -1), tf.float32)

    return tf.einsum('bik,bjk->bij', row_mask, col_mask)

def weighted_sum(weight, values):
    '''Calcualte the weighted sum.

    Args:
        weight: a tensor with shape [batch, time, dimension]
        values: a tensor with shape [batch, dimension, values_dimension]

    Return:
        a tensor with shape [batch, time, values_dimension]

    Raises:
    '''
    return tf.einsum('bij,bjk->bik', weight, values)

def layer_norm_debug(x, axis = None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    if axis is None:
        axis = [-1]
    shape = [x.shape[i] for i in axis]

    scale = tf.get_variable(
        name='scale',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    mean = tf.reduce_mean(x, axis=axis, keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keep_dims=True)
    norm = (x-mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias

def FFN(x, out_dimension_0=None, out_dimension_1=None):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    with tf.variable_scope('FFN_1'):
        y = dense(x, out_dimension_0)
        y = tf.nn.relu(y)
    with tf.variable_scope('FFN_2'):
        z = dense(y, out_dimension_1) #, add_bias=False)  #!!!!
    return z

def dense(x, out_dimension=None, add_bias=True):
    '''Add dense connected layer, Wx + b.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Return:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    if out_dimension is None:
        out_dimension = x.shape[-1]

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], out_dimension],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    if add_bias:
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        return tf.einsum('bik,kj->bij', x, W) + bias
    else:
        return tf.einsum('bik,kj->bij', x, W)
