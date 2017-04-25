# variational autoencoder
# @author: luyi edited from https://jmetzen.github.io/2015-11-27/vae.html
import tensorflow as tf
import numpy as np
import os.path
from glob import glob
import scipy.misc
import math

flags = tf.app.flags
flags.DEFINE_integer("epochs", 3, "Epoch to train [25]")
flags.DEFINE_integer("input_h", 64, "input height of the image")
flags.DEFINE_integer("input_w", 64, "input width of the image")
flags.DEFINE_integer("channel", 3, "num of image channels")
flags.DEFINE_integer("z_dim", 100, "dimension of the latent vector")
flags.DEFINE_integer("mode", 0, "mlp:0, conv:1")
flags.DEFINE_float("learning_rate", 0.00005, "learning rate")
flags.DEFINE_string("input_dir", "data", "path to images")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("display_step", 1, "display step")
flags.DEFINE_string("sample_dir", "sample_out", "samples during training")
flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint directory")
flags.DEFINE_integer("save_step", 1000, "save model every 1000 steps")
flags.DEFINE_string("log_dir", "summary", "training summary")
flags.DEFINE_string("transfer", "sigmoid", "transfer function")

FLAGS = flags.FLAGS

checkpoint_dir = "{}_{}_{}".format(FLAGS.ckpt_dir, FLAGS.mode, FLAGS.transfer)
sample_dir = os.path.join(checkpoint_dir, FLAGS.sample_dir)
log_dir = os.path.join(checkpoint_dir, FLAGS.log_dir)
latent_size = FLAGS.z_dim
input_size = FLAGS.input_h * FLAGS.input_w * FLAGS.channel
batch_size = FLAGS.batch_size
x = tf.placeholder(tf.float32, [None, FLAGS.input_h * FLAGS.input_w * FLAGS.channel], name="x")
z = tf.placeholder(tf.float32, [None, latent_size], name="z")
k_h = k_w = 5
d_h = d_w = 2
n_f = 64

num_sample = 9

if FLAGS.transfer == "sigmoid":
    transfer = tf.nn.sigmoid
elif FLAGS.transfer == "relu":
    transfer = tf.nn.relu
else:
    transfer = tf.nn.softplus


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def transform(image, resize_height=64, resize_width=64):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 255.


def inverse_transform(images):
    return (images + 1.) / 2.


def get_image(image_path, resize_height=64, resize_width=64, is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return transform(image, resize_height, resize_width)


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def conv_init(shape):
    return tf.random_normal(shape, stddev=0.02)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def mlp_encoder(x, reuse=False):
    layer1_size = 500
    layer2_size = 500
    with tf.variable_scope("mlp_encoder") as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable(name="w1", initializer=xavier_init(input_size, layer1_size))
        b1 = tf.get_variable(name="b1", initializer=tf.zeros(layer1_size))
        w2 = tf.get_variable(name="w2", initializer=xavier_init(layer1_size, layer2_size))
        b2 = tf.get_variable(name="b2", initializer=tf.zeros(layer2_size))
        w_mean = tf.get_variable(name="w_mean", initializer=xavier_init(layer2_size, latent_size))
        b_mean = tf.get_variable(name="b_mean", initializer=tf.zeros(latent_size))
        w_logvar = tf.get_variable(name="w_logvar", initializer=xavier_init(layer2_size, latent_size))
        b_logvar = tf.get_variable(name="b_logvar", initializer=tf.zeros(latent_size))
        layer1 = transfer(tf.nn.xw_plus_b(x, w1, b1), name="layer1")
        layer2 = transfer(tf.nn.xw_plus_b(layer1, w2, b2), name="layer2")
        z_mean = tf.nn.xw_plus_b(layer2, w_mean, b_mean, name="z_mean")
        z_logvar = tf.nn.xw_plus_b(layer2, w_logvar, b_logvar, name="z_logvar")
        return z_mean, z_logvar


def mlp_decoder(encoding, reuse=False):
    layer1_size = 500
    layer2_size = 500
    with tf.variable_scope("mlp_decoder") as scope:
        if reuse:
            scope.reuse_variables()
        w_mean = tf.get_variable(name="w_mean", initializer=xavier_init(latent_size, layer1_size))
        b_mean = tf.get_variable(name="b_mean", initializer=tf.zeros(layer1_size))
        w2 = tf.get_variable(name="w2", initializer=xavier_init(layer1_size, layer2_size))
        b2 = tf.get_variable(name="b2", initializer=tf.zeros(layer2_size))
        w1 = tf.get_variable(name="w1", initializer=xavier_init(layer2_size, input_size))
        b1 = tf.get_variable(name="b1", initializer=tf.zeros(input_size))
        layer1 = transfer(tf.matmul(encoding, w_mean) + b_mean, name="layer1")
        layer2 = transfer(tf.matmul(layer1, w2) + b2, name="layer2")
        mean_out = tf.sigmoid(tf.matmul(layer2, w1) + b1, name="mean_out")
        return mean_out


#TO DO: pooling? batch normalization?
def conv_encoder(x, reuse=False):
    with tf.variable_scope("conv_encoder"):
        if reuse:
            scope.reuse_variables()
        x = tf.reshape(x, [-1, FLAGS.input_h, FLAGS.input_w, FLAGS.channel])
        w1 = tf.get_variable('w1', initializer=conv_init([k_h, k_w, FLAGS.channel, n_f]))
        b1 = tf.get_variable('b1', [n_f], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable('w2', initializer=conv_init([k_h, k_w, n_f, n_f * 2]))
        b2 = tf.get_variable('b2', [n_f * 2], initializer=tf.constant_initializer(0.0))
        w3 = tf.get_variable('w3', initializer=conv_init([k_h, k_w, n_f * 2, n_f * 4]))
        b3 = tf.get_variable('b3', [n_f * 4], initializer=tf.constant_initializer(0.0))
        w4 = tf.get_variable('w4', initializer=conv_init([k_h, k_w, n_f * 4, n_f * 8]))
        b4 = tf.get_variable('b4', [n_f * 8], initializer=tf.constant_initializer(0.0))
        conv1 = transfer(tf.nn.conv2d(x, w1, strides=[1, d_h, d_w, 1], padding='SAME', name="conv1") + b1)
        conv2 = transfer(tf.nn.conv2d(conv1, w2, strides=[1, d_h, d_w, 1], padding='SAME', name="conv2") + b2)
        conv3 = transfer(tf.nn.conv2d(conv2, w3, strides=[1, d_h, d_w, 1], padding='SAME', name="conv3") + b3)
        conv4 = transfer(tf.nn.conv2d(conv3, w4, strides=[1, d_h, d_w, 1], padding='SAME', name="conv4") + b4)
        _, conv4_h, conv4_h, conv4_nf = conv4.get_shape().as_list()
        flat_size = conv4_h * conv4_h * conv4_nf
        flat = tf.reshape(conv4, [-1, flat_size], name="flat")
        w_mean = tf.get_variable(name="w_mean", initializer=xavier_init(flat_size, latent_size))
        b_mean = tf.get_variable(name="b_mean", initializer=tf.zeros(latent_size))
        w_logvar = tf.get_variable(name="w_logvar", initializer=xavier_init(flat_size, latent_size))
        b_logvar = tf.get_variable(name="b_logvar", initializer=tf.zeros(latent_size))
        z_mean = tf.nn.xw_plus_b(flat, w_mean, b_mean, name="z_mean")
        z_logvar = tf.nn.xw_plus_b(flat, w_logvar, b_logvar, name="z_logvar")
        return z_mean, z_logvar


def conv_decoder(encoding, reuse=False):
    with tf.variable_scope("conv_decoder") as scope:
        if reuse:
            scope.reuse_variables()
        s_h, s_w = FLAGS.input_h, FLAGS.input_w
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        batch_size = tf.shape(encoding)[0]

        w1 = tf.get_variable('w1', initializer=conv_init([k_h, k_w, FLAGS.channel, n_f]))
        b1 = tf.get_variable('b1', [FLAGS.channel], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable('w2', initializer=conv_init([k_h, k_w, n_f, n_f * 2]))
        b2 = tf.get_variable('b2', [n_f], initializer=tf.constant_initializer(0.0))
        w3 = tf.get_variable('w3', initializer=conv_init([k_h, k_w, n_f * 2, n_f * 4]))
        b3 = tf.get_variable('b3', [n_f * 2], initializer=tf.constant_initializer(0.0))
        w4 = tf.get_variable('w4', initializer=conv_init([k_h, k_w, n_f * 4, n_f * 8]))
        b4 = tf.get_variable('b4', [n_f * 4], initializer=tf.constant_initializer(0.0))
        flat_size = s_h16 * s_w16 * n_f * 8
        w_mean = tf.get_variable(name="w_mean", initializer=xavier_init(latent_size, flat_size))
        b_mean = tf.get_variable(name="b_mean", initializer=tf.zeros(flat_size))
        flat = tf.nn.xw_plus_b(encoding, w_mean, b_mean, name="flat")
        conv4 = tf.reshape(flat, [-1, s_h16, s_w16, n_f * 8], name="conv4")
        conv3 = transfer(tf.nn.conv2d_transpose(conv4, w4, output_shape=[batch_size, s_h8, s_w8, n_f * 4],
                                                strides=[1, d_h, d_w, 1]) + b4,
                         name="conv3")
        conv2 = transfer(tf.nn.conv2d_transpose(conv3, w3, output_shape=[batch_size, s_h4, s_w4, n_f * 2],
                                                strides=[1, d_h, d_w, 1]) + b3,
                         name="conv2")
        conv1 = transfer(tf.nn.conv2d_transpose(conv2, w2, output_shape=[batch_size, s_h2, s_w2, n_f],
                                                strides=[1, d_h, d_w, 1]) + b2,
                         name="conv1")
        mean_out = tf.nn.sigmoid(
            tf.nn.conv2d_transpose(conv1, w1, output_shape=[batch_size, s_h, s_w, FLAGS.channel],
                                   strides=[1, d_h, d_w, 1]) + b1, name="mean_out")
        return tf.reshape(mean_out, [batch_size, -1])


def generate(encoding, mode):
    if mode == 0:
        return mlp_decoder(encoding, reuse=True)
    else:
        return conv_decoder(encoding, reuse=True)


data = glob(os.path.join(FLAGS.input_dir, "*.jpg"))
n_samples = len(data)
print "Total number of images: ", n_samples


def train():
    sess = tf.InteractiveSession()
    latent_repr = np.random.normal(size=(num_sample, latent_size)).astype(np.float)
    if FLAGS.mode == 0:
        z_mean, z_logvar = mlp_encoder(x)
        std = tf.sqrt(tf.exp(z_logvar), name="z_std")
        eps = tf.random_normal((batch_size, latent_size), 0, 1,
                               dtype=tf.float32)
        encoding = z_mean + std * eps
        mean_out = mlp_decoder(encoding)
    else:
        z_mean, z_logvar = conv_encoder(x)
        std = tf.sqrt(tf.exp(z_logvar), name="z_std")
        eps = tf.random_normal((batch_size, latent_size), 0, 1,
                               dtype=tf.float32)
        encoding = z_mean + std * eps
        mean_out = conv_decoder(encoding)
    reconstr_loss = \
        -tf.reduce_mean(-0.5 * (tf.square(mean_out - x)))
    sample = generate(z, mode=FLAGS.mode)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_logvar
                                       - tf.square(z_mean)
                                       - tf.exp(z_logvar))
    cost = reconstr_loss + latent_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
    global_step = tf.get_variable(name="global_step", initializer=0)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    with tf.name_scope("summary") as scope:
        cost_sum = tf.summary.scalar("cost", cost)
        latent_loss_sum = tf.summary.scalar("latent_loss", tf.reduce_mean(latent_loss))
        reconstr_loss_sum = tf.summary.scalar("reconstr_loss", tf.reduce_mean(reconstr_loss))
        G_sum = tf.summary.image("G", tf.reshape(sample, [num_sample, FLAGS.input_h, FLAGS.input_w, FLAGS.channel]))

    summary_op = tf.summary.merge([cost_sum, latent_loss_sum, reconstr_loss_sum])
    print("------Training start------")
    if load(saver, checkpoint_dir, sess):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
    # for var in vars: print var
    total_batch = int(n_samples / batch_size)
    current_step = sess.run(global_step)
    current_epoch = current_step // total_batch
    writer.add_session_log(session_log=tf.SessionLog(status=tf.SessionLog.START), global_step=current_step)
    for epoch in range(current_epoch, FLAGS.epochs):
        # Loop over all batches
        for i in range(current_step % total_batch, total_batch):
            incr_global_step = tf.assign(global_step, global_step + 1)
            start = i * (batch_size)
            end = min(n_samples, (i + 1) * batch_size)
            data_batch = data[start:end]
            batch_xs = [get_image(img, resize_height=FLAGS.input_h, resize_width=FLAGS.input_w,
                                  is_grayscale=False) for img in data_batch]
            batch_xs = np.array(batch_xs).astype(np.float32)
            batch_xs = batch_xs.reshape(end - start, -1)
            # Fit training using batch data
            current_step, cost_scalar, summary, _, _ = sess.run(
                [global_step, cost, summary_op, optimizer, incr_global_step], feed_dict={x: batch_xs})

            writer.add_summary(summary, global_step=current_step)
            # Display logs per epoch step
            if (current_step) % FLAGS.display_step == 0:
                print("Epoch: %04d" % (epoch + 1), "step: %06d" % (current_step), "cost={:0.9f}".format(cost_scalar))
                if not os.path.exists(sample_dir):
                    os.mkdir(sample_dir)
                sample_img = sess.run(sample, feed_dict={z: latent_repr})
                sample_img = sample_img.reshape(num_sample, FLAGS.input_h, FLAGS.input_w, FLAGS.channel)
                imsave(sample_img, [int(math.sqrt(num_sample)), int(math.sqrt(num_sample))],
                       os.path.join(sample_dir, '{}.jpg'.format(current_step)))
                writer.add_summary(sess.run(G_sum, feed_dict={z: latent_repr}), global_step=current_step)

            if (current_step) % FLAGS.save_step == 0:
                save(saver, checkpoint_dir, sess, current_step)


def imsave(images, size, path):
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    return scipy.misc.imsave(path, merge(images, size))


def save(saver, checkpoint_dir, sess, step):
    model_name = "model-{}".format(step)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name),
               global_step=step)
    print(" [*] Success to save at step {}".format(step))


def load(saver, checkpoint_dir, sess):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False


train()
