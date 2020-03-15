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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


LOCATION_OF_DATA = "/home/nat/nat_wmt14/"


_ENDE_TRAIN_DATASETS = [
    [
        "",  # pylint: disable=line-too-long
        ("train/train.en",
         "train/train.de")
    ],
]

_ENDE_TEST_DATASETS = [
    [
        "",  # pylint: disable=line-too-long
        ("dev/valid.en",
         "dev/valid.de")
    ],
]

@registry.register_problem
class TranslateEndeWmt32k(translate.TranslateProblem):
  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_name(self):
    return "vocab.ende"

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        _ENDE_TRAIN_DATASETS)
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_ende_tok_%s" % tag)
    return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK
