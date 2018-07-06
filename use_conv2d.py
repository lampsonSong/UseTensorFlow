import tensorflow as tf 
import tensorflow.contrib.slim as slim

x1   = tf.constant([
        [  
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]]
        ]
    ],shape=[1,3,3,1])

x2   = tf.constant([
        [  
            [[1.0], [2.0], [3.0], [3.0]],
            [[4.0], [5.0], [6.0], [6.0]],
            [[7.0], [8.0], [9.0], [9.0]],
            [[7.0], [8.0], [9.0], [9.0]]
        ]
    ],shape=[1,4,4,1])

w = tf.constant([
        [
            [1.0],  [1.0], [1.0],
            [1.0],  [1.0], [1.0],
            [1.0], [1.0],[1.0]

        ]
    ],shape=[3,3,1,1])

y1 = tf.nn.conv2d(x1, w, strides=[1, 2, 2, 1], padding='SAME')
y2 = slim.conv2d(x1, 1, [3, 3], weights_initializer=tf.ones_initializer, stride = 2, padding='SAME') 

y1_4 = tf.nn.conv2d(x2, w, strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    y1_value, y2_value, x1_value = sess.run([y1,y2,x1]) 
    print("shapes are", y1_value.shape) 
    print(y1_value) 
    print("y2 shape: ", y2_value.shape)
    print(y2_value)
    print("----------")
    y1_4_value = sess.run(y1_4)
    print(y1_4_value)
