# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils.dump_hidden import dump_hidden

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

from tensor2tensor.models import transformer as original_transformer

@registry.register_model
class TransformerNatClWord(t2t_model.T2TModel):
  """Attention net.  See file docstring."""
  def __init__(self,
               hparams,
               mode,
               problem_hparams,
               problem_idx=0,
               data_parallelism=None,
               ps_devices=None,
               decode_hparams=None):
    self._input_modality_setting = problem_hparams.input_modality
    self._target_modality_setting = problem_hparams.target_modality
    super().__init__(hparams,
                     mode,
                     problem_hparams,
                     problem_idx,
                     data_parallelism,
                     ps_devices,
                     decode_hparams)
    if hparams.teacher_model:
      self._teacher_input_modality = registry.create_modality(
          self._input_modality_setting["inputs"], hparams)
      self._teacher_target_modality = registry.create_modality(
          self._target_modality_setting, hparams)
  def encode(self, inputs, target_space, hparams):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, hidden_dim]
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(encoder_input, self_attention_bias,
                                         hparams)

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             return_attention_weight=False):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    additional_outputs = None

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        return_attention_weight=return_attention_weight)

    if isinstance(decoder_output, tuple):
      decoder_output, additional_outputs = decoder_output

    decoder_output = dump_hidden(decoder_output, "decoder_output.cosine")

    if additional_outputs is not None:
      return tf.expand_dims(decoder_output, axis=2), additional_outputs
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)

  def teacher_model(self, features, return_attention_weight=False):
    hparams = self._hparams
    # tf.logging.info(repr(features))
    additional_outputs = None
    input_modality = self._teacher_input_modality
    target_modality = self._teacher_target_modality
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom(features["inputs_raw"])

    with tf.variable_scope(target_modality.name):
      targets = target_modality.bottom(features["targets_raw"])

    with tf.variable_scope("body"):
      inputs = common_layers.flatten4d3d(inputs)
      target_space = features["target_space_id"]

      (encoder_input, encoder_self_attention_bias,
       encoder_decoder_attention_bias) = (
        original_transformer.transformer_prepare_encoder(
          inputs, target_space, hparams))

      encoder_output = original_transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams)

      targets = common_layers.flatten4d3d(targets)

      decoder_input, decoder_self_attention_bias = (
        original_transformer.transformer_prepare_decoder(
        targets, hparams))
      decoder_output = original_transformer.transformer_decoder(
          decoder_input,
          encoder_output,
          decoder_self_attention_bias,
          encoder_decoder_attention_bias,
          hparams,
          return_attention_weight=return_attention_weight
      )
      if isinstance(decoder_output, tuple):
        decoder_output, additional_outputs = decoder_output
      decoder_output = tf.expand_dims(decoder_output, axis=2)
    with tf.variable_scope(target_modality.name):
      logits = target_modality.top(decoder_output, features["targets_raw"])
    return decoder_output, logits, additional_outputs

  def model_fn_body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    return_attention_weight = hparams.distill_encdec > 0.0

    if hparams.teacher_model:
      with tf.variable_scope("teacher_model"):
        (teacher_hidden, teacher_logits, teacher_additional_outputs) = (
          self.teacher_model(features,
                             return_attention_weight=return_attention_weight))
        teacher_hidden = tf.stop_gradient(teacher_hidden)
        teacher_logits = tf.stop_gradient(teacher_logits)

    inputs = features.get("inputs")
    encoder_output, encoder_decoder_attention_bias = (None, None)
    if inputs is not None:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams)

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)
    (decoder_input, decoder_self_attention_bias, encoder_decoder_attention_bias,
     targets_padding) = transformer_prepare_decoder(
        inputs, targets, encoder_output, encoder_decoder_attention_bias,
        hparams)

    decoded_hidden = self.decode(decoder_input, encoder_output,
                                 encoder_decoder_attention_bias,
                                 decoder_self_attention_bias, hparams,
                                 return_attention_weight
                                 =return_attention_weight)

    return decoded_hidden

  def _greedy_infer(self, features, decode_length):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
       samples: [batch_size, input_length + decode_length]
       logits: Not returned
       losses: Not returned

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    decoded_ids, _ = self._fast_decode(features, decode_length)
    return decoded_ids, None, None

  def _beam_decode(self, features, decode_length, beam_size, top_beams,
                   alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    decoded_ids, scores = self._fast_decode(
        features, decode_length, beam_size, top_beams, alpha)
    return {"outputs": decoded_ids, "scores": scores}

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    decode_hparams = self._decode_hparams
    inputs = features["inputs"]
    targets = features["targets"]
    target_modality = self._problem_hparams.target_modality
    batch_size = tf.shape(inputs)[0]
    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    raw_inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(raw_inputs, dp)
    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode, inputs, features["target_space_id"], hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if beam_size > 1:  # Beam Search
      # we will search the sentence with length in
      # (input_len - beam_size + length_bias,
      #  input_len + beam_size + length_bias)
      assert hparams.teacher_model
      with tf.variable_scope("body"):
        with tf.variable_scope("teacher_model"):
          teacher_input_modality = self._teacher_input_modality
          with tf.variable_scope(teacher_input_modality.name):
            teacher_inputs = teacher_input_modality.bottom_sharded(raw_inputs,
                                                                   dp)
          with tf.variable_scope("body"):
            teacher_inputs = dp(common_layers.flatten4d3d, teacher_inputs)
            (teacher_encoder_input, teacher_encoder_self_attention_bias,
             teacher_encoder_decoder_attention_bias) = dp(
              original_transformer.transformer_prepare_encoder,
              teacher_inputs, features["target_space_id"], hparams)
            teacher_encoder_output = dp(
                original_transformer.transformer_encoder, teacher_encoder_input,
                teacher_encoder_self_attention_bias, hparams)
      teacher_encoder_output = teacher_encoder_output[0]
      teacher_encoder_decoder_attention_bias = (
        teacher_encoder_decoder_attention_bias[0])

      n_copies = 2 * beam_size - 1
      inputs = dp(copy_batches, inputs, n_copies)
      encoder_output = copy_batches(encoder_output, n_copies)
      encoder_decoder_attention_bias = copy_batches(
          encoder_decoder_attention_bias, n_copies)
      teacher_encoder_output = copy_batches(teacher_encoder_output, n_copies)
      teacher_encoder_decoder_attention_bias = copy_batches(
          teacher_encoder_decoder_attention_bias, n_copies)

      length_bias = tf.range(-beam_size + 1 + decode_hparams.length_bias,
                             beam_size + decode_hparams.length_bias)
      length_bias = tf.tile(length_bias, (batch_size, ))
      with tf.variable_scope("body"):
        (decoder_input, decoder_self_attention_bias,
         encoder_decoder_attention_bias, targets_padding) = dp(
          transformer_prepare_decoder, inputs, None, encoder_output,
          encoder_decoder_attention_bias, hparams,
          True, length_bias)
        body_outputs = dp(self.decode, decoder_input, encoder_output,
                          encoder_decoder_attention_bias,
                          decoder_self_attention_bias, hparams)
      targets_padding = targets_padding[0]

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]
      logits = tf.squeeze(logits, axis=[2, 3])
      temperature = (0.0 if hparams.sampling_method == "argmax"
                     else hparams.sampling_temp)
      decoded_ids = common_layers.sample_with_temperature(logits, temperature)
      with tf.variable_scope("body"):
        with tf.variable_scope("teacher_model"):
          modified_decoded_ids = tf.expand_dims(decoded_ids, axis=-1)
          modified_decoded_ids = tf.expand_dims(modified_decoded_ids, axis=1)
          if len(modified_decoded_ids.shape) < 5:
            modified_decoded_ids = tf.expand_dims(modified_decoded_ids, axis=4)
          s = tf.shape(modified_decoded_ids)
          modified_decoded_ids = tf.reshape(modified_decoded_ids,
                                            [s[0] * s[1], s[2], s[3], s[4]])
          teacher_target_modality = self._teacher_target_modality
          with tf.variable_scope(teacher_target_modality.name):
            teacher_targets = teacher_target_modality.targets_bottom(
                modified_decoded_ids)
          teacher_targets = common_layers.flatten4d3d(teacher_targets)
          with tf.variable_scope("body"):
            teacher_decoder_input, teacher_decoder_self_attention_bias = (
              original_transformer.transformer_prepare_decoder(
                  teacher_targets, hparams))
            teacher_decoder_output = original_transformer.transformer_decoder(
                teacher_decoder_input,
                teacher_encoder_output,
                teacher_decoder_self_attention_bias,
                teacher_encoder_decoder_attention_bias,
                hparams,
            )
            teacher_decoder_output = tf.expand_dims(teacher_decoder_output,
                                                    axis=2)
          with tf.variable_scope(teacher_target_modality.name):
            teacher_logits = teacher_target_modality.top(
                teacher_decoder_output, None)
          teacher_logits = tf.squeeze(teacher_logits, axis=[2, 3])
      log_probs = beam_search.log_prob_from_logits(teacher_logits)
      max_decode_length = tf.shape(decoded_ids)[1]
      batch_id = tf.reshape(tf.range(batch_size * n_copies),
                            (batch_size * n_copies, 1, 1))
      batch_id = tf.tile(batch_id, (1, max_decode_length, 1))
      time_step_id = tf.reshape(tf.range(max_decode_length),
                                (1, max_decode_length, 1))
      time_step_id = tf.tile(time_step_id, (batch_size * n_copies, 1, 1))
      gather_indices = tf.concat([batch_id,
                                  time_step_id,
                                  tf.expand_dims(tf.to_int32(decoded_ids),
                                                 axis=-1)],
                                 axis=-1)
      pos_log_probs = tf.gather_nd(log_probs, gather_indices)
      pos_log_probs *= (1.0 - targets_padding)
      sentence_log_probs = tf.reduce_sum(pos_log_probs, axis=-1)
      sentence_log_probs = tf.reshape(sentence_log_probs,
                                      (batch_size, n_copies))
      if decode_hparams.alpha > 0.0:
        targets_length = tf.reduce_sum(1.0 - targets_padding, axis=-1)
        length_penalty = ((5 + targets_length) ** decode_hparams.alpha
                          / (6 ** decode_hparams.alpha))
        length_penalty = tf.reshape(length_penalty, (batch_size, n_copies))
        sentence_log_probs = sentence_log_probs / length_penalty
      max_prob_sentece_id = tf.argmax(sentence_log_probs, axis=1)
      scores = tf.reduce_max(sentence_log_probs, axis=1)
      decoded_ids = tf.gather(decoded_ids,
                              tf.range(batch_size) * n_copies
                              + tf.to_int32(max_prob_sentece_id))

    else:  # Greedy
      targets = tf.expand_dims(targets, axis=1)
      if len(targets.shape) < 5:
        targets = tf.expand_dims(targets, axis=4)
      s = tf.shape(targets)
      targets = tf.reshape(targets, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)
      targets = dp(common_layers.flatten4d3d, targets)
      with tf.variable_scope("body"):
        (decoder_input, decoder_self_attention_bias,
         encoder_decoder_attention_bias, targets_padding) = dp(
          transformer_prepare_decoder, inputs, targets, encoder_output,
          encoder_decoder_attention_bias, hparams,
          decode_hparams.use_inputs_length, decode_hparams.length_bias)
        body_outputs = dp(self.decode, decoder_input, encoder_output,
                          encoder_decoder_attention_bias,
                          decoder_self_attention_bias, hparams)
      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]
      logits = tf.squeeze(logits, axis=[2, 3])
      temperature = (0.0 if hparams.sampling_method == "argmax"
                     else hparams.sampling_temp)
      decoded_ids = common_layers.sample_with_temperature(logits, temperature)
      scores = None

    return decoded_ids, scores

def copy_batches(x, num_copies):
  """
  :param x: a tensor with shape [batch_size, ...]
  :param num_copies: an integer
  :return: a tensor with shape [batch_size * num_copies, ...]
    e.g. x = [1, 2], the output will be [1, 1, 2, 2]
  """
  x_shape = tf.shape(x)
  x = tf.reshape(x, (x_shape[0], 1, -1))
  x = tf.tile(x, (1, num_copies, 1))
  x = tf.reshape(x, tf.concat([[-1], x_shape[1:]], axis=0))
  return x

def transformer_prepare_encoder(inputs, target_space, hparams):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        tf.shape(inputs)[1])
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding")
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space

  if hparams.pos == "timing":
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def attention_bias_without_self(length):
  return tf.reshape(-1e9 * tf.eye(length), [1, 1, length, length])


def transformer_prepare_decoder(inputs, targets, encoder_output,
                                encoder_decoder_attention_bias,
                                hparams, use_inputs_length=False,
                                length_bias=0):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """
  inputs = common_layers.flatten4d3d(inputs)
  inputs_padding = common_attention.embedding_to_padding(inputs)
  inputs_length = tf.reduce_sum(1.0 - inputs_padding, axis=-1)
  batch_size = tf.shape(inputs)[0]
  max_inputs_length = tf.shape(inputs)[1]
  inputs_index = tf.to_float(tf.range(max_inputs_length))

  if use_inputs_length:
    targets_length = tf.maximum(inputs_length + tf.to_float(length_bias), 1)
    max_targets_length = tf.to_int32(tf.reduce_max(targets_length))
    targets_index = tf.to_float(tf.range(max_targets_length))
    batched_index = tf.tile(tf.expand_dims(targets_index, axis=0),
                            (batch_size, 1))
    targets_padding = tf.to_float(batched_index >=
                                  tf.expand_dims(targets_length, 1))
  else:
    targets_padding = common_attention.embedding_to_padding(targets)
    targets_length = tf.reduce_sum(1.0 - targets_padding, axis=-1)
    max_targets_length = tf.shape(targets)[1]
    targets_index = tf.to_float(tf.range(max_targets_length))

  step = inputs_length / targets_length

  closest_encoder_postion = (tf.expand_dims(step, axis=-1) * targets_index)
  closest_encoder_id = tf.to_int32(tf.round(closest_encoder_postion))
  closest_encoder_id = tf.reshape(closest_encoder_id,
                                  (batch_size, max_targets_length, 1))
  batch_id = tf.reshape(tf.range(batch_size), (batch_size, 1, 1))
  batch_id = tf.tile(batch_id, (1, max_targets_length, 1))
  gather_indices = tf.concat([batch_id, closest_encoder_id], axis=-1)

  if hparams.use_cl:
    # CL for NAT, with increasing probability:
    # 1. Replace AT decoder input to NAT input with word-to-word substitute
    # 2. Expand AT attention bias to all words
    nat_decoder_input = tf.gather_nd(inputs, gather_indices)
    at_decoder_input = common_layers.shift_right_3d(targets)
    now_length = tf.shape(at_decoder_input)[1]

    # NAT self-attention bias
    ignore_padding = common_attention.attention_bias_ignore_padding(
      targets_padding)
    nat_decoder_self_attention_bias = ignore_padding
    if hparams.attention_without_self:
        nat_decoder_self_attention_bias += attention_bias_without_self(
          max_targets_length)
    # AT self-attention bias
    at_decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
    at_decoder_self_attention_bias = tf.tile(at_decoder_self_attention_bias, (batch_size, 1, 1, 1))

    global_step = tf.to_float(tf.train.get_or_create_global_step())
    decoder_self_attention_bias = tf.cond(global_step < hparams.start_nat_attention_bias_step,
      lambda: at_decoder_self_attention_bias,
      lambda: nat_decoder_self_attention_bias
      )

    def return_no_sub():
      return -0.5, 0.0

    def return_sub_prob_and_len():
      if hparams.fix_rate:
        sub_prob = tf.to_float(hparams.fix_rate_to)
      elif hparams.linear_rate:
        sub_prob = ((global_step - hparams.at_pretrain_steps + 10000)/10000) / hparams.total_steps
        sub_prob = tf.minimum(sub_prob, 1.0)
      elif hparams.ladder_rate:
        sub_prob = tf.to_float(tf.to_int32(((global_step - hparams.at_pretrain_steps + 10000)/10000) / 20)*20) /hparams.total_steps
        sub_prob = tf.minimum(sub_prob, 1.0)
      elif hparams.invlog_rate:
        used_step = tf.maximum(hparams.total_steps - ((global_step - hparams.at_pretrain_steps+10000)/10000), 1.0)
        sub_prob = 1.0 - tf.log(used_step) / tf.log(hparams.total_steps)
        sub_prob = tf.minimum(sub_prob, 1.0)
      else:
        sub_prob = tf.log((global_step - hparams.at_pretrain_steps + 10000)/10000) / tf.log(hparams.total_steps)
        sub_prob = tf.minimum(sub_prob, 1.0)
      return sub_prob, tf.round(sub_prob * tf.to_float(now_length))

    cl_prob, sub_len = tf.cond(global_step - hparams.at_pretrain_steps < 0,
      return_no_sub,
      return_sub_prob_and_len
      )
    sub_len = tf.to_int32(sub_len)
    cl_prob = tf.cond(tf.equal(tf.mod(global_step, 100), 0),
      lambda: tf.Print(cl_prob, [global_step, cl_prob], summarize=1000, message="cl prob"),
      lambda: cl_prob
      )
    sub_len = tf.cond(tf.equal(tf.mod(global_step, 100), 0),
      lambda: tf.Print(sub_len, [global_step, sub_len], summarize=1000, message="sub_len"),
      lambda: sub_len
      )

    def at_cond():
      return at_decoder_input

    def nat_cond():
      # Execute word substitute instead of directly use nat_decoder_input as input
      all_index = tf.random_shuffle(tf.range(now_length))
      random_index = all_index[:sub_len]
      len_eye_matrix = tf.eye(now_length)
      select_eye_matrix = tf.gather(len_eye_matrix, random_index, axis=0) # (sub_len, now_length)
      select_eye_vector = tf.reduce_sum(select_eye_matrix, axis=0) # (now_length)
      select_batch_mask = tf.tile(tf.expand_dims(tf.expand_dims(select_eye_vector, axis=-1), axis=0),
              (batch_size, 1, 1))
      altered_decoder_input = select_batch_mask*nat_decoder_input + (1-select_batch_mask)*at_decoder_input
      return altered_decoder_input

    def mix_mask():
      all_index = tf.random_shuffle(tf.range(now_length))
      random_index = all_index[:sub_len]
      len_eye_matrix = tf.eye(now_length)
      select_eye_matrix = tf.gather(len_eye_matrix, random_index, axis=0) # (sub_len, now_length)
      select_eye_vector = tf.reduce_sum(select_eye_matrix, axis=0) # (now_length)
      select_batch_mask = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(select_eye_vector, axis=-1), axis=0), axis=0),
              (batch_size, 1, 1, 1)) # (bs, 1, now_len, 1)
      mixed_self_mask = select_batch_mask*nat_decoder_self_attention_bias + (1-select_batch_mask)*at_decoder_self_attention_bias
      return mixed_self_mask

    if hparams.substitute_mask:
      decoder_self_attention_bias = mix_mask()

    if hparams.previous_rate_change:
      decoder_input = tf.cond(
        tf.less(tf.random_uniform([]), cl_prob),
        nat_cond,
        at_cond
        )
    else:
      decoder_input = nat_cond()

  else:
    decoder_input = tf.gather_nd(inputs, gather_indices)
    ignore_padding = common_attention.attention_bias_ignore_padding(
      targets_padding)
    decoder_self_attention_bias = ignore_padding
    if hparams.attention_without_self:
        decoder_self_attention_bias += attention_bias_without_self(
          max_targets_length)

  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)

  return (decoder_input, decoder_self_attention_bias,
          encoder_decoder_attention_bias, targets_padding)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = encoder_input
  with tf.variable_scope(name):
    pad_remover = None
    if hparams.use_pad_remover:
      pad_remover = expert_utils.PadRemover(
          common_attention.attention_bias_to_padding(
              encoder_self_attention_bias))
    for layer in xrange(hparams.num_encoder_layers or
                        hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams, pad_remover)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)

def compute_positional_qkv(query_antecedent, value_antecedent, total_key_depth,
                           total_value_depth, q_filter_width=1, kv_filter_width=1,
                           q_padding="VALID", kv_padding="VALID"):
  """Computes query, key and value.

    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels]
      total_key_depth: an integer
      total_value_depth: and integer
      q_filter_width: An integer specifying how wide you want the query to be.
      kv_filter_width: An integer specifying how wide you want the keys and values
      to be.
      q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

    Returns:
      q, k, v : [batch, length, depth] tensors
    """
  if q_filter_width == kv_filter_width == 1:
    # self attention with single position q, k, and v
    combined = common_layers.conv1d(
      query_antecedent,
      total_key_depth * 2,
      1,
      name="qk_transform")
    q, k = tf.split(
      combined, [total_key_depth, total_key_depth],
      axis=2)
    v = common_layers.conv1d(
      value_antecedent,
      total_value_depth,
      1,
      name="v_transform")
    return q, k, v

  # self attention
  q = common_layers.conv1d(
    query_antecedent,
    total_key_depth,
    q_filter_width,
    padding=q_padding,
    name="q_transform")
  k = common_layers.conv1d(
    query_antecedent,
    total_key_depth,
    kv_filter_width,
    padding=kv_padding,
    name="k_transform")
  v = common_layers.conv1d(
    value_antecedent,
    total_key_depth,
    kv_filter_width,
    padding=kv_padding,
    name="v_transform")
  return q, k, v

def multihead_positional_attention(query_antecedent,
                                   value_antecedent,
                                   bias,
                                   total_key_depth,
                                   total_value_depth,
                                   output_depth,
                                   num_heads,
                                   dropout_rate,
                                   max_relative_position=None,
                                   image_shapes=None,
                                   attention_type="dot_product",
                                   block_length=128,
                                   block_width=128,
                                   q_filter_width=1,
                                   kv_filter_width=1,
                                   q_padding="VALID",
                                   kv_padding="VALID",
                                   cache=None,
                                   gap_size=0,
                                   num_memory_blocks=2,
                                   name=None,
                                   **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d" or any attention function with the
                    signature (query, key, value, **kwargs)
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hiddem_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_positional_attention",
      values=[query_antecedent, value_antecedent]):
    q, k, v = compute_positional_qkv(query_antecedent, value_antecedent, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)

    if cache is not None:
      if attention_type != "dot_product":
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes)
    elif attention_type == "dot_product_relative":
      x = common_attention.dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                         dropout_rate, image_shapes)
    elif attention_type == "local_mask_right":
      x = common_attention.masked_local_attention_1d(q, k, v, block_length=block_length)
    elif attention_type == "local_unmasked":
      x = common_attention.local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = common_attention.masked_dilated_self_attention_1d(q, k, v, block_length,
                                           block_width,
                                           gap_size,
                                           num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = common_attention.dilated_self_attention_1d(q, k, v, block_length,
                                    block_width,
                                    gap_size,
                                    num_memory_blocks)
    x = common_attention.combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder",
                        return_attention_weight=False):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string

  Returns:
    y: a Tensors
  """
  x = decoder_input
  positional_embedding = common_attention.get_timing_signal_1d(
    tf.shape(x)[1], hparams.hidden_size)
  x = dump_hidden(x, 'decoder_input.cosine')
  additional_outputs = {}
  if return_attention_weight:
    additional_outputs["attention_weight"] = []
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None

      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_layers.layer_preprocess(x, hparams)
          y = dump_hidden(y, layer_name + '.self_att.pre.cosine')
          y = common_attention.multihead_attention(
              y,
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache)
          y = dump_hidden(y, layer_name + '.self_att.core.cosine')
          x = common_layers.layer_postprocess(x, y, hparams)
          x = dump_hidden(x, layer_name + '.self_att.post.cosine')
        if hparams.positional_attention:
          with tf.variable_scope("positional_attention"):
            y = common_layers.layer_preprocess(x, hparams)
            y = dump_hidden(y, layer_name + '.pos_att.pre.cosine')
            y = multihead_positional_attention(
              positional_embedding,
              y,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache)
            y = dump_hidden(y, layer_name + '.pos_att.core.cosine')
            x = common_layers.layer_postprocess(x, y, hparams)
            x = dump_hidden(x, layer_name + '.pos_att.post.cosine')
        if encoder_output is not None and hparams.encdec_attention:
          with tf.variable_scope("encdec_attention"):
            y = common_layers.layer_preprocess(x, hparams)
            y = dump_hidden(y, layer_name + '.encdec_att.pre.cosine')
            y = common_attention.multihead_attention(
                y, encoder_output, encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size, hparams.num_heads,
                hparams.attention_dropout,
                return_attention_weight=return_attention_weight)
            if return_attention_weight:
              y, attention_weight = y
              additional_outputs["attention_weight"].append(attention_weight)
            y = dump_hidden(y, layer_name + '.encdec_att.core.cosine')
            x = common_layers.layer_postprocess(x, y, hparams)
            x = dump_hidden(x, layer_name + '.encdec_att.post.cosine')
        with tf.variable_scope("ffn"):
          y = common_layers.layer_preprocess(x, hparams)
          y = dump_hidden(y, layer_name + '.ffn.pre.cosine')
          y = transformer_ffn_layer(y, hparams)
          y = dump_hidden(y, layer_name + '.ffn.core.cosine')
          x = common_layers.layer_postprocess(x, y, hparams)
          x = dump_hidden(x, layer_name + '.ffn.post.cosine')
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    if additional_outputs != {}:
      return common_layers.layer_preprocess(x, hparams), additional_outputs
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams, pad_remover=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  if hparams.ffn_layer == "conv_hidden_relu":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = tf.shape(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif hparams.ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.filter_size, hparams.num_heads,
        hparams.attention_dropout)
  elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  else:
    assert hparams.ffn_layer == "none"
    return x

from tensor2tensor.models.transformer import (transformer_small,
                                              transformer_deep_small,
                                              transformer_deep_small_iwslt16,
                                              transformer_base_v1,
                                              transformer_base_v2)

def add_nat_hparmas(hparams):
  hparams.add_hparam("positional_attention", True)
  hparams.add_hparam("attention_without_self", True)
  hparams.add_hparam("encdec_attention", True)
  hparams.add_hparam("decoder_input_type", "hard_uniform")
  hparams.add_hparam("attention_coverage", "no_regularization")
  hparams.add_hparam("attention_coverage_alpha", 0.0)
  hparams.add_hparam("regularize_hidden", 0.0)
  hparams.add_hparam("regularize_hidden_normalized", True)
  hparams.add_hparam("regularize_hidden_pos", "neighbor")
  hparams.add_hparam("regularize_hidden_hinge_param", -1.0)
  hparams.add_hparam("regularize_hidden_loss", "identity")
  hparams.add_hparam("teacher_model", False)
  hparams.add_hparam("teacher_num_hidden_layers", 5)
  hparams.add_hparam("distill_encdec", 0.0)

  hparams.add_hparam("use_cl", True)
  hparams.add_hparam("at_pretrain_steps", 10000)
  hparams.add_hparam("start_nat_attention_bias_step", 1000000)
  hparams.add_hparam("total_steps", 200.0)
  hparams.add_hparam("fix_rate", False)
  hparams.add_hparam("fix_rate_to", 1.0)
  hparams.add_hparam("linear_rate", False)
  hparams.add_hparam("ladder_rate", False)
  hparams.add_hparam("invlog_rate", False)
  hparams.add_hparam("substitute_mask", False)
  hparams.add_hparam("previous_rate_change", True)
  return hparams

@registry.register_hparams
def transformer_nat_cl_base_v1():
  hparams = transformer_base_v1()
  hparams = add_nat_hparmas(hparams)
  return hparams

