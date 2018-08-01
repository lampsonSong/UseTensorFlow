# created by lampson.song @ 2018-08-01
# show weights / inter_resutls from frozed pb file

# The used mobilenet_v2_1 model is downloaded from:
# https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz

import tensorflow as tf
import numpy as np

restored_graph_def = tf.GraphDef()

def show_weights(node_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for node in restored_graph_def.node:
            if node.name == node_name:
                weights = tf.convert_to_tensor(node.attr['value'].tensor)
                sess.run(tf.Print(weights,[weights],summarize=9)) # summarize is the number shown weights

def show_inter_results(input_layer_name, inter_layer_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        graph = tf.get_default_graph()
        input_layer = graph.get_tensor_by_name(input_layer_name)
        inter_layer = graph.get_tensor_by_name(inter_layer_name)

        input_data = np.ones((224,224,3))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(inter_layer, feed_dict={input_layer:[input_data]})
            print(out)

if __name__ == '__main__':
    graph = tf.get_default_graph()

    frozen_graph = "/home/lampson/2T_disk/model/tensorflow/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb"

    with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
        
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    # list all node names
    for n in restored_graph_def.node: 
        print(n.name)

    show_weights("Const_145")
    show_inter_results("input:0","Const_100:0")
