# created by lampson.song @ 2018-08-01
# show weights from frozed pb file

# The used mobilenet_v2_1 model is downloaded from:
# https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz

import tensorflow as tf

restored_graph_def = tf.GraphDef()

def show_weights(node_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for node in restored_graph_def.node:
            if node.name == node_name:
                print(node)
                weights = tf.convert_to_tensor(n.attr['value'].tensor)
                print(weights.shape)
                sess.run(tf.Print(weights,[weights],summarize=9)) # summarize is the number shown weights

if __name__ == '__main__':
    graph = tf.get_default_graph()

    frozen_graph = "./mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb"

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
