import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        # input
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)       
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.int64)
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 * self.batch_size
        )
        return images, depths, invalid_depths


def output_predict(depths, images, output_dir):
    print("output predict into %s" % len(output_dir))
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)
