import tensorflow as tf
import numpy as np

flags = tf.flags
flags.DEFINE_string("hidden_state_dir", "", "Filename to dump hidden states")
FLAGS = flags.FLAGS

_HIDDEN_COUNTER = 0


def dump_hidden(x, name, dump_measure="cosine"):
  """

  :param x: [batchsize = 1, len, hidden_dim]
  :param name: a name added to filename
  :param dump_measure: "cosine", "l2_norm"
  :return:
  """
  global _HIDDEN_COUNTER
  if FLAGS.hidden_state_dir != "":
    x_norm = tf.norm(x, ord=2, axis=-1)
    normalized_x = tf.nn.l2_normalize(x, -1)
    cosine_similarity = tf.reduce_sum(
      tf.expand_dims(normalized_x, axis=1)
      * tf.expand_dims(normalized_x, axis=2), axis=-1)
    file_name = '%s%.3d.%s' % (FLAGS.hidden_state_dir, _HIDDEN_COUNTER, name)

    def dump(x, norm, cos):
      np.set_printoptions(threshold=np.nan)
      if dump_measure == 'cosine':
        tf.logging.info("dump " + file_name + '.csv')
        np.savetxt(file_name + '.csv', cos[0], fmt='%.2f', delimiter=', ')
      if dump_measure == 'l2_norm':
        tf.logging.info("dump " + file_name + '.csv')
        np.savetxt(file_name + '.csv', norm[0], fmt='%.2f', delimiter=', ')
      if dump_measure == 'attention':
        # for i, w in enumerate(x[0]):
        tf.logging.info("dump " + file_name + '.csv')
        np.savetxt(file_name + '.csv', x[0], fmt='%.6f', delimiter=', ')
      return np.array(0.0, np.float32)

    _HIDDEN_COUNTER += 1

    with tf.control_dependencies(
        [tf.py_func(dump, (x, x_norm, cosine_similarity), tf.float32)]):
      x = tf.identity(x)
  return x
