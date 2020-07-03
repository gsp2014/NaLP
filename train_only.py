import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import pickle
from log import Logger
from batching import *
from model import NaLP

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "WikiPeople", "The sub data dir.")
tf.flags.DEFINE_string("dataset_name", "WikiPeople", "The name of the dataset.")
tf.flags.DEFINE_string("wholeset_name", "WikiPeople_permutate", "Name of the whole dataset for negative sampling or computing the filtered metrics.")
tf.flags.DEFINE_string("model_name", 'WikiPeople', "")
tf.flags.DEFINE_integer("embedding_dim", 100, "The embedding dimension.")
tf.flags.DEFINE_integer("n_filters", 200, "The number of filters.")
tf.flags.DEFINE_integer("n_gFCN", 1200, "The number of hidden units of fully-connected layer in g-FCN.")
tf.flags.DEFINE_integer("batch_size", 128, "The batch size.")
tf.flags.DEFINE_boolean("is_trainable", True, "")
tf.flags.DEFINE_float("learning_rate", 0.00005, "The learning rate.")
tf.flags.DEFINE_integer("n_epochs", 5000, "The number of training epochs.")
tf.flags.DEFINE_boolean("if_restart", False, "")
tf.flags.DEFINE_integer("start_epoch", 0, "Change this when restarting")
tf.flags.DEFINE_integer("saveStep", 100, "Save the model every saveStep")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("run_folder", "./", "The dir to store models.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# The log file to store the parameters and the training details of each epoch
logger = Logger('logs', 'run_'+FLAGS.model_name+'_'+str(FLAGS.embedding_dim)+'_'+str(FLAGS.n_filters)+'_'+str(FLAGS.n_gFCN)+'_'+str(FLAGS.batch_size)+'_'+str(FLAGS.learning_rate)).logger
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))

# Load training data
logger.info("Loading data...")
afolder = FLAGS.data_dir + '/'
if FLAGS.sub_dir != '':
    afolder = FLAGS.data_dir + '/' + FLAGS.sub_dir + '/'
with open(afolder + FLAGS.dataset_name + ".bin", 'rb') as fin:
    data_info = pickle.load(fin)
train = data_info["train_facts"]
values_indexes = data_info['values_indexes']
roles_indexes = data_info['roles_indexes']
role_val = data_info['role_val']
value_array = np.array(list(values_indexes.values()))
role_array = np.array(list(roles_indexes.values()))

# Load the whole dataset for negative sampling in "batching.py"
with open(afolder + FLAGS.wholeset_name + ".bin", 'rb') as fin:
    data_info1 = pickle.load(fin)
whole_train = data_info1["train_facts"]
logger.info("Loading data... finished!")

with tf.Graph().as_default():
    tf.set_random_seed(1234)
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        aNaLP = NaLP(
            n_values=len(values_indexes),
            n_roles=len(roles_indexes),
            embedding_dim=FLAGS.embedding_dim,
            n_filters=FLAGS.n_filters,
            n_gFCN=FLAGS.n_gFCN,
            batch_size=FLAGS.batch_size*2,
            is_trainable=FLAGS.is_trainable)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(aNaLP.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        
        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", FLAGS.model_name))
        logger.info("Writing to {}\n".format(out_dir))

        # Train Summaries
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
   
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, arity):
            """
            A single training step
            """
            feed_dict = {
              aNaLP.input_x: x_batch,
              aNaLP.input_y: y_batch,
              aNaLP.arity: arity
            }
            _, loss = sess.run([train_op, aNaLP.loss], feed_dict)
            return loss
        
        # If restart, then load the model
        if FLAGS.if_restart == True:
            _file = checkpoint_prefix + "-" + str(FLAGS.start_epoch)
            aNaLP.saver.restore(sess, _file)

        # Training
        n_batches_per_epoch = []
        for i in train:
            ll = len(i)
            if ll == 0:
                n_batches_per_epoch.append(0)
            else:
                n_batches_per_epoch.append(int((ll - 1) / FLAGS.batch_size) + 1)
        for epoch in range(FLAGS.start_epoch, FLAGS.n_epochs):
            train_loss = 0
            for i in range(len(train)):
                train_batch_indexes = np.array(list(train[i].keys())).astype(np.int32)
                train_batch_values = np.array(list(train[i].values())).astype(np.float32)
                for batch_num in range(n_batches_per_epoch[i]):
                    arity = i + 2  # 2-ary in index 0
                    x_batch, y_batch = Batch_Loader(train_batch_indexes, train_batch_values, values_indexes, roles_indexes, role_val, FLAGS.batch_size, arity, whole_train[i])
                    tmp_loss = train_step(x_batch, y_batch, arity)
                    train_loss = train_loss + tmp_loss
                
            logger.info("nepoch: "+str(epoch+1)+", trainloss: "+str(train_loss))
            if (epoch+1) % FLAGS.saveStep == 0:
                path = aNaLP.saver.save(sess, checkpoint_prefix, global_step=epoch+1)
                logger.info("Saved model checkpoint to {}\n".format(path))
        train_summary_writer.close
