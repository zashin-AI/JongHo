import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    shape_result = [dynamic[i] if s is None else s for i, s in enumerate(static)]
    print("shape_result \t", shape_result)
    return shape_result

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    sfm = ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
    print("softmax result \t", sfm)
    return sfm

def gelu(x):
    g_elu = 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
    print("g_elu \t", g_elu)
    return g_elu

def gelu2(x):
    g_elu_2 = x * tf.sigmoid(1.702 * x)
    print("g_elu_2 \t", g_elu_2)
    return g_elu_2

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[axis].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
        x = x * tf.rsqrt(s + epsilon)
        x = x*g
        print("norm result \t", x)
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    split_state_result = tf.reshape(x, start + [n, m//n])
    print("split_state_result \t", split_state_result)
    return split_state_result

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    merge_result = tf.reshape(x, start + [a*b])
    print("merge_result \t", merge_result)
    return merge_result

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])), start+[nf])
        print("conv1d layer \t", c)
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def attn(x, scope, n_state, *, past, hparams):
    print("=====att_n_state", n_state, "\t||\tatt_n_head", hparams.n_head, "=====") # att_n_state 512 att_n_head 8
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        print(" mask_attn_weights \t", w)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        if not hparams.bert:
            w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        print("multihead_attn \t", a)
        return a

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)

        wk = tf.get_variable("k_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wq = tf.get_variable("q_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wv = tf.get_variable("v_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        k = tf.einsum("bsf,hef->bhse", x, wk)
        q = tf.einsum("bsf,hef->bhse", x, wq)
        v = tf.einsum("bsf,hef->bhse", x, wv)

        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        wc = tf.get_variable("c_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state*hparams.n_layer)))
        a = tf.einsum("bhse,hef->bsf", a, wc)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu2(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model_summary(model):
    model_vars = tf.trainable_variables()
    model.model_analyzer.analyze_vars(model_vars, print_info=True)

def model(hparams, X, Y=None, past=None, scope='model', reuse=False):
    print("===========================================")
    print("hparams \t",hparams) # n_ctx=1024,n_embd=512,n_head=8,n_layer=24,n_vocab=512,bert=False,bert_mask_prob=0.15,clf=False
    print("X \t", X)            # Tensor("split:0", shape=(8, 1024), dtype=int32)
    print("Y \t", Y)            # Tensor("split_1:0", shape=(8, 1000), dtype=float32)
    print("past \t", past)      # None
    print("scope \t", scope)    # model
    print("reuse \t", reuse)    # False
    print("===========================================")

    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if hparams.bert:
            M = tf.greater(tf.random.uniform([batch, sequence]), hparams.bert_mask_prob)
            M = tf.cast(M, tf.float32)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        wtet = tf.get_variable('wtet', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.0))
        past_length = 0 if past is None else tf.shape(past)[-2]

        h = tf.gather(wte, X)

        if hparams.bert:
            h = h * tf.expand_dims(M, 2)
        else:
            sos = tf.get_variable('sos', [hparams.n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            sos_tok = tf.ones([batch, 1, hparams.n_embd], dtype=tf.float32) * sos
            h = tf.concat([sos_tok, h[:,:-1,:]], axis=1)

        h += tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            # print("what is h \t", h)
            # print("what is present \t", present)
            # print("=========================")
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        # h = norm(h, 'ln_f')
        # print("what is norm \t", h)
        # print("what is present \t", present)
        # print("what is presents >>> \t", presents)

        # model_summary(h)

        # Generative loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        gen_logits = tf.matmul(h_flat, wtet, transpose_b=True)
        gen_logits = tf.reshape(gen_logits, [batch, sequence, hparams.n_vocab])
        results['gen_logits'] = gen_logits
        print("gen_logits \t", gen_logits)

        gen_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits, labels=X)
        print("softmax : \t", gen_losses)
        if hparams.bert:
            IM = 1.0 - M
            gen_losses = tf.reduce_sum(gen_losses * IM, axis=1) / tf.reduce_sum(IM, axis=1)
            results['gen_loss'] = tf.reduce_mean(gen_losses)
        else:
            results['gen_loss'] = tf.reduce_mean(gen_losses)
        print("mean softmax \t" ,  gen_losses)

        # Classification loss.
        with tf.variable_scope('clf', reuse=reuse):
            classes = shape_list(Y)[1]
            if hparams.clf:
                wclf = tf.get_variable('wclf', [classes, hparams.n_embd],
                                      initializer=tf.random_normal_initializer(stddev=0.0))
            else:
                wclf = tf.zeros([classes, hparams.n_embd], dtype=tf.float32)

        h = tf.reduce_mean(h, axis=1)  # average pool over sequence
        clf_logits = tf.matmul(h, wclf, transpose_b=True)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=clf_logits, labels=Y)
        results['clf_loss'] = tf.reduce_mean(clf_losses)
        print("===clf_losses===\t", results['clf_loss'])

        correct = tf.equal(tf.argmax(clf_logits, -1), tf.argmax(Y, -1))
        results['accuracy'] = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0
        print("===correct===\t", results['accuracy'])

        return results

"""
shape_result     [8, 1024]
WARNING:tensorflow:From c:\image-gpt\src\nmb_model.py:183: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

WARNING:tensorflow:From c:\image-gpt\src\nmb_model.py:45: The name tf.rsqrt is deprecated. Please use tf.math.rsqrt instead.

norm result      Tensor("model/h0/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h0/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h0/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h0/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h0/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h0/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h0/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h0/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h1/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h1/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h1/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h1/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h1/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h1/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h1/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h1/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h2/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h2/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h2/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h2/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h2/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h2/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h2/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h2/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h3/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h3/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h3/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h3/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h3/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h3/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h3/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h3/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h4/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h4/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h4/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h4/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h4/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h4/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h4/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h4/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h5/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h5/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h5/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h5/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h5/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h5/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h5/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h5/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h6/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h6/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h6/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h6/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h6/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h6/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h6/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h6/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h7/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h7/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h7/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h7/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h7/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h7/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h7/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h7/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h8/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h8/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h8/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h8/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h8/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h8/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h8/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h8/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h9/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h9/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h9/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h9/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h9/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h9/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h9/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h9/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h10/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h10/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h10/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h10/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h10/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h10/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h10/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h10/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h11/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h11/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h11/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h11/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h11/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h11/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h11/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h11/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h12/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h12/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h12/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h12/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h12/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h12/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h12/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h12/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h13/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h13/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h13/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h13/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h13/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h13/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h13/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h13/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h14/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h14/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h14/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h14/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h14/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h14/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h14/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h14/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h15/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h15/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h15/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h15/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h15/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h15/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h15/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h15/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h16/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h16/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h16/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h16/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h16/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h16/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h16/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h16/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h17/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h17/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h17/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h17/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h17/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h17/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h17/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h17/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h18/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h18/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h18/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h18/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h18/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h18/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h18/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h18/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h19/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h19/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h19/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h19/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h19/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h19/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h19/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h19/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h20/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h20/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h20/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h20/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h20/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h20/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h20/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h20/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h21/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h21/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h21/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h21/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h21/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h21/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h21/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h21/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h22/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h22/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h22/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h22/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h22/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h22/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h22/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h22/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h23/ln_1/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
shape_result     [8, 8, 1024, 1024]
 mask_attn_weights       Tensor("model/h23/attn/sub_2:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
softmax result   Tensor("model/h23/attn/truediv:0", shape=(8, 8, 1024, 1024), dtype=float32, device=/device:GPU:0)
multihead_attn   Tensor("model/h23/attn/MatMul_1:0", shape=(8, 8, 1024, 64), dtype=float32, device=/device:GPU:0)
norm result      Tensor("model/h23/ln_2/mul_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 512]
conv1d layer     Tensor("model/h23/mlp/c_fc/Reshape_2:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
g_elu_2          Tensor("model/h23/mlp/mul_1:0", shape=(8, 1024, 2048), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1024, 2048]
conv1d layer     Tensor("model/h23/mlp/c_proj/Reshape_2:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
gen_logits       Tensor("model/Reshape_1:0", shape=(8, 1024, 512), dtype=float32, device=/device:GPU:0)
softmax :        Tensor("model/SparseSoftmaxCrossEntropyWithLogits/Reshape_2:0", shape=(8, 1024), dtype=float32, device=/device:GPU:0)
mean softmax     Tensor("model/SparseSoftmaxCrossEntropyWithLogits/Reshape_2:0", shape=(8, 1024), dtype=float32, device=/device:GPU:0)
shape_result     [8, 1000]
===clf_losses===         Tensor("model/Mean_2:0", shape=(), dtype=float32, device=/device:GPU:0)
===correct===    Tensor("model/mul_1:0", shape=(), dtype=float32, device=/device:GPU:0)
"""