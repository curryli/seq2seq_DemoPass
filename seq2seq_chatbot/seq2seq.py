from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection=None):

    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        # 对输出概率进行归一化和取log，这样序列概率相乘就可以变成概率相加
        probs = tf.log(tf.nn.softmax(prev))
        if i == 1:
            probs = tf.reshape(probs[0, :], [-1, num_symbols])
        if i > 1:
            # 将当前序列的概率与之前序列概率相加得到结果之前有beam_szie个序列，本次产生num_symbols个结果，
            # 所以reshape成这样的tensor
            probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])
        # 选出概率最大的前beam_size个序列,从beam_size * num_symbols个元素中选出beam_size个
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        # beam_size * num_symbols，看对应的是哪个序列和单词
        symbols = indices % num_symbols  # Which word in vocabulary.
        beam_parent = indices // num_symbols  # Which hypothesis it came from.
        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)

        # 对beam-search选出的beam size个单词进行embedding，得到相应的词向量
        emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        emb_prev = tf.reshape(emb_prev, [-1, embedding_size])
        return emb_prev

    return loop_function


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, beam_search=True, beam_size=10):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
        embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        loop_function = _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection)
        return beam_attention_decoder(
            emb_inp, initial_state, attention_states, cell, embedding, output_size=output_size,
            num_heads=num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention, output_projection=output_projection,
            beam_size=beam_size)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, beam_search=True, beam_size=10):
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(encoder_cell, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention, beam_search=beam_search, beam_size=beam_size)

