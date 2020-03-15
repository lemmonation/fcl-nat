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

"""Script to average values of variables in a list of checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import os
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("student", "",
                    "student model checkpoint.")
flags.DEFINE_string("teacher", "",
                    "teacher model checkpoint.")
flags.DEFINE_string("tgt", "",
                    "tgt model checkpoint.")
flags.DEFINE_string("prefix_student", "",
                    "Prefix (e.g., directory) to student checkpoint.")
flags.DEFINE_string("prefix_teacher", "",
                    "Prefix (e.g., directory) to teacher checkpoint.")
flags.DEFINE_string("prefix_tgt", "",
                    "Prefix (e.g., directory) to tgt checkpoint.")
flags.DEFINE_integer("step", None, 
                    "global step of output checkpoint.")
flags.DEFINE_string("output_path", "/tmp/averaged.ckpt",
                    "Path to output the merged checkpoint to.")


def checkpoint_exists(path):
  return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
          tf.gfile.Exists(path + ".index"))


def main(_):
  # Get the checkpoints list from flags and run some basic checks.
  student = FLAGS.student
  teacher = FLAGS.teacher
  tgt = FLAGS.tgt
  step = int(student[11:]) if FLAGS.step is None else FLAGS.step
  print("step =", step)
  if FLAGS.prefix_student:
    student = os.path.join(FLAGS.prefix_student, student)
  if FLAGS.prefix_teacher:
    teacher = os.path.join(FLAGS.prefix_teacher, teacher)
  if FLAGS.prefix_tgt:
    tgt = os.path.join(FLAGS.prefix_tgt, tgt)
 
  checkpoints = [student, teacher, tgt]

  # Read variables from all checkpoints and merge them.
  tf.logging.info("Reading variables and merging checkpoints:")
  var_list = tf.contrib.framework.list_variables(tgt)
  var_values, var_dtypes = {}, {}
  tf.logging.info("Reading variables from %s" % tgt)
  for (name, shape) in var_list:
    if not name.startswith("global_step"):
      var_values[name] = np.zeros(shape)

  reader_student = tf.contrib.framework.load_checkpoint(student)
  reader_teacher = tf.contrib.framework.load_checkpoint(teacher)
  reader_tgt = tf.contrib.framework.load_checkpoint(tgt)
  tf.logging.info("Copying all the variables")
  for name in var_values:
    if name.startswith('transformer_nart/body/teacher_model'):
      look_name = 'transformer' + name[len('transformer_nart/body/teacher_model'):]
      tensor = reader_teacher.get_tensor(look_name)
    else:
      try:
        look_name = name
        tensor = reader_student.get_tensor(look_name)
      except:
        tf.logging.info("%s not found in student model, copy from tgt model" % name)
        print("%s not found in student model, copy from tgt model" % name)
        tensor = reader_tgt.get_tensor(name)
    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor
  
  tf_vars = [
      tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
      for v in var_values
  ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(
      step, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, FLAGS.output_path, global_step=global_step)

  tf.logging.info("Copied checkpoint saved in %s", FLAGS.output_path)


if __name__ == "__main__":
  tf.app.run()
