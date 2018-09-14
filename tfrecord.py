import random
import tensorflow as tf
from PIL import Image

def create_record(records_path, data_path, img_txt):
    writer = tf.python_io.TFRecordWriter(records_path)
    
    img_list = []
    with open(img_txt, 'r') as fr:
        img_list = fr.readlines()
    random.shuffle(img_list)
    cnt = 0
    # get image info
    for img_info in img_list:
        # img path 
        img_name = img_info.split(' ')[0]
        # label
        img_cls = int(img_info.split(' ')[1])
        img_path = data_path + img_name
        img = Image.open(img_path)
        
        # preprocess, if required
        img = img.resize((128, 128))
        img_raw = img.tobytes()

        # write the label and value
        example = tf.train.Example(
           features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_cls])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
           }))
        # write information
        writer.write(example.SerializeToString())
        # output logs
        cnt += 1
        if cnt % 1000 == 0:
            print "processed %d images" % cnt
    writer.close()

def read_record(filename):
    f_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(f_queue)

    features = tf.parse_single_example(serialized_example, features={'label':tf.FixedLenFeature([], tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    return img, label


# where to put the tfrecords
records_path = 'imgs/test.tfrecords'

# img path and label txt
data_path = 'imgs/'
img_txt = 'imgs/list.txt'
create_record(records_path, data_path, img_txt)

read_record(records_path)
