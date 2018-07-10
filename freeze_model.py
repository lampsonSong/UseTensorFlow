# created by lampson.song @ 2018-7-10
# get a pb file from meta

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

model_dir = './'

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    saver = tf.train.import_meta_graph(model_dir+'model-1000.meta', clear_devices=True)
    saver.restore(sess, model_dir+'model-1000')

    graph = convert_variables_to_constants(sess, sess.graph_def, ['here_is_output_ops_name'])
    tf.train.write_graph(graph, '.', 'graph.pb', as_text=True)
