import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import pickle
from multiprocessing import JoinableQueue, Queue, Process
from log import Logger
from batching import *
from model import NaLP

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "WikiPeople", "The sub data dir.")
tf.flags.DEFINE_string("dataset_name", "WikiPeople", "The name of the dataset.")
tf.flags.DEFINE_string("wholeset_name", "WikiPeople_permutate", "Name of the whole dataset for computing the filtered metrics.")
tf.flags.DEFINE_string("model_name", 'WikiPeople', "")
tf.flags.DEFINE_integer("embedding_dim", 100, "The embedding dimension.")
tf.flags.DEFINE_integer("n_filters", 200, "The number of filters.")
tf.flags.DEFINE_integer("n_gFCN", 1200, "The number of hidden units of fully-connected layer in g-FCN.")
tf.flags.DEFINE_integer("batch_size", 128, "The batch size.")
tf.flags.DEFINE_boolean("is_trainable", False, "")
tf.flags.DEFINE_float("learning_rate", 0.00005, "The learning rate.")
tf.flags.DEFINE_integer("n_epochs", 5000, "The number of training epochs.")
tf.flags.DEFINE_boolean("if_restart", False, "")
tf.flags.DEFINE_integer("start_epoch", 0, "Change this when restarting")
tf.flags.DEFINE_integer("evalStep", 100, "Evaluate the model every saveStep")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("metric_num", 4, "")
tf.flags.DEFINE_integer("valid_or_test", 1, "validate: 1, test: 2")
tf.flags.DEFINE_string("gpu_ids", "0,1,2,3", "Comma-separated gpu id")
tf.flags.DEFINE_string("run_folder", "./", "The dir to store models.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# The log file to store the parameters and the evaluation details of each epoch
logger = Logger('logs', str(FLAGS.valid_or_test)+'_evalres_'+FLAGS.model_name+'_'+str(FLAGS.embedding_dim)+'_'+str(FLAGS.n_filters)+'_'+str(FLAGS.n_gFCN)+'_'+str(FLAGS.batch_size)+'_'+str(FLAGS.learning_rate)).logger
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))
gpu_ids = list(map(int, FLAGS.gpu_ids.split(",")))

# Load validation and test data
logger.info("Loading data...")
afolder = FLAGS.data_dir + '/'
if FLAGS.sub_dir != '':
    afolder = FLAGS.data_dir + '/' + FLAGS.sub_dir + '/'
with open(afolder + FLAGS.dataset_name + ".bin", 'rb') as fin:
    data_info = pickle.load(fin)
valid = data_info["valid_facts"]
test = data_info['test_facts']
values_indexes = data_info['values_indexes']
roles_indexes = data_info['roles_indexes']
role_val = data_info['role_val']
value_array = np.array(list(values_indexes.values()))
role_array = np.array(list(roles_indexes.values()))

# Load the whole dataset for computing the filtered metrics
with open(afolder + FLAGS.wholeset_name + ".bin", 'rb') as fin:
    data_info1 = pickle.load(fin)
whole_train = data_info1["train_facts"]
whole_valid = data_info1["valid_facts"]
whole_test = data_info1['test_facts']
logger.info("Loading data... finished!")

# Prepare validation and test facts
x_valid = []
y_valid = []
for k in valid:
    x_valid.append(np.array(list(k.keys())).astype(np.int32))
    y_valid.append(np.array(list(k.values())).astype(np.float32))
x_test = []         
y_test = []         
for k in test:      
    x_test.append(np.array(list(k.keys())).astype(np.int32))
    y_test.append(np.array(list(k.values())).astype(np.int32))

# Output directory for models and checkpoint directory
out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", FLAGS.model_name))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class Predictor(Process):
    """
    Predictor for evaluation
    """
    def __init__(self, in_queue, out_queue, epoch, gpu_id):
        Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.epoch = epoch
        self.gpu_id = gpu_id
    def run(self):
        # set GPU id before importing tensorflow!
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_id)

        # import tensorflow here
        import tensorflow as tf
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        aNaLP = NaLP(
            n_values=len(values_indexes),
            n_roles=len(roles_indexes),
            embedding_dim=FLAGS.embedding_dim,
            n_filters=FLAGS.n_filters,
            n_gFCN=FLAGS.n_gFCN,
            batch_size=FLAGS.batch_size,
            is_trainable=FLAGS.is_trainable)

        _file = checkpoint_prefix + "-" + str(self.epoch)
        aNaLP.saver.restore(sess, _file)

        while True:
            dat = self.in_queue.get()
            if dat is None:
                self.in_queue.task_done()
                break
            else:
                (x_batch, y_batch, arity, ind) = dat
                feed_dict = {
                aNaLP.input_x: x_batch,
                aNaLP.input_y: y_batch,
                aNaLP.arity: arity,
                }
                scores, loss = sess.run([aNaLP.predictions, aNaLP.loss], feed_dict)
                self.out_queue.put((scores, loss, ind))
                self.in_queue.task_done()
        sess.close()
        return

def eval_one(x_batch, y_batch, evaluation_queue, result_queue, data_index, pred_ind=0):
    """
    Predict the pred_ind-th element (value/role) of each fact in x_batch
    """
    mrr = 0.0
    hits1 = 0.0
    hits3 = 0.0
    hits10 = 0.0
    total_loss = 0.0
    for i in range(len(x_batch)):
        if pred_ind % 2 == 0:  # predict role
            tmp_array = role_array
            right_index = np.argwhere(role_array == x_batch[i][pred_ind])[0][0]
        else:
            tmp_array = value_array  #predict value
            right_index = np.argwhere(value_array == x_batch[i][pred_ind])[0][0]
        new_x_batch = np.tile(x_batch[i], (len(tmp_array), 1))
        new_x_batch[:, pred_ind] = tmp_array
        new_y_batch = np.tile(np.array([-1]).astype(np.int32), (len(tmp_array), 1))
        new_y_batch[right_index] = [1]
        while len(new_x_batch) % FLAGS.batch_size != 0:
            new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
            new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)
        tmp_array1 = new_x_batch[:, pred_ind]
        listIndexes = range(0, len(new_x_batch), FLAGS.batch_size)
        nn = len(listIndexes)
        results = []
        tmp_res_list = []
        for tmpIndex in range(nn):
            tmp_res_list.append([])
        arity = int(len(x_batch[i])/2)
        for tmpIndex in range(nn - 1):
            evaluation_queue.put((new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]], new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]], arity, tmpIndex))
        evaluation_queue.put((new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:], arity, nn-1))
        evaluation_queue.join()
        
        for tmp_id in range(nn):
            (res, loss, ind) = result_queue.get()
            tmp_res_list[ind] = res
            total_loss = total_loss + loss
        for tmp_id in range(nn):
            results = np.append(results, tmp_res_list[tmp_id])

        results = np.reshape(results, [tmp_array1.shape[0], 1])
        results_with_id = np.hstack(
            (np.reshape(tmp_array1, [tmp_array1.shape[0], 1]), results))
        results_with_id = results_with_id[np.argsort(-results_with_id[:, 1])]
        results_with_id = results_with_id[:, 0].astype(int)
        _filter = 0
        for tmpxx in results_with_id:
            if tmpxx == x_batch[i][pred_ind]:
                break
            tmp_list = list(x_batch[i])
            tmp_list[pred_ind] = tmpxx
            tmpTriple = tuple(tmp_list)
            if (len(whole_train) > data_index) and (tmpTriple in whole_train[data_index]):
                continue
            elif (len(whole_valid) > data_index) and (tmpTriple in whole_valid[data_index]):
                continue
            elif (len(whole_test) > data_index) and (tmpTriple in whole_test[data_index]):
                continue
            else:
                _filter += 1

        mrr += 1.0 / (_filter + 1)
        if _filter < 10:
            hits10 += 1
            if _filter < 3:
                hits3 += 1
                if _filter < 1:
                    hits1 += 1
    return np.array([total_loss, mrr, hits10, hits3, hits1])

def eval_all(epoch, x_test, y_test, evaluation_queue, result_queue):
    """
    Predict all the elements (values and roles) of each fact in the whole set x_test
    """
    role_results = np.zeros(FLAGS.metric_num)
    val_results = np.zeros(FLAGS.metric_num)
    role_c = 0
    val_c = 0
    all_loss = 0.0
    len_data = 0
    for i in range(len(x_test)):
        if len(x_test[i]) == 0:
            continue
        len_data = len_data + len(x_test[i])
        n_ary = i + 2  # 2-ary in index 0
        if epoch == FLAGS.n_epochs: 
            for j in range(2*n_ary):
                tmp = eval_one(x_test[i], y_test[i], evaluation_queue, result_queue, i, j)
                tmp_results = tmp[1:]
                all_loss = all_loss + tmp[0]
                if j % 2 == 0:
                    role_results = role_results + tmp_results
                    role_c = role_c + len(x_test[i])
                else:
                    val_results = val_results + tmp_results
                    val_c = val_c + len(x_test[i])
        else:  # If it is not the last epoch, only predict values 
            for j in range(2*n_ary):
                if j % 2 == 0:
                    continue
                tmp = eval_one(x_test[i], y_test[i], evaluation_queue, result_queue, i, j)
                tmp_results = tmp[1:]
                all_loss = all_loss + tmp[0]
                val_results = val_results + tmp_results
                val_c = val_c + len(x_test[i])

    for i in range(len(gpu_ids)):
        evaluation_queue.put(None)
    logger.info(FLAGS.dataset_name+", len(data): "+str(len_data))
    logger.info("epoch: "+str(epoch)+", testloss: "+str(all_loss/(role_c+val_c)))
    if role_c == 0:
        role_c = 1
    logger.info("epoch: "+str(epoch)+", role_value: "+str(role_results/role_c)+'; '+str(val_results/val_c))

def check_epoch_finish(model_dir, epoch):
    """
    Check if the epoch training finishes 
    """
    for root, dirs, files in os.walk(model_dir):
        for name in files:
            if name.find(str(epoch)+'.') != -1:
                return True
    return False

if __name__ == "__main__":
    cur_epoch = FLAGS.start_epoch
    while True:
        if check_epoch_finish(checkpoint_dir, cur_epoch) == True:
            logger.info("begin eval"+str(cur_epoch))
            evaluation_queue = JoinableQueue()
            result_queue = Queue()
            p_list = []
            for i in range(len(gpu_ids)):
                p = Predictor(evaluation_queue, result_queue, cur_epoch, gpu_ids[i])
                p_list.append(p)
            for p in p_list:
                p.start()
            if FLAGS.valid_or_test == 1:
                eval_all(cur_epoch, x_valid, y_valid, evaluation_queue, result_queue)
            else:
                eval_all(cur_epoch, x_test, y_test, evaluation_queue, result_queue)
            for p in p_list:
                p.join()
            logger.info("finish eval"+str(cur_epoch))
            cur_epoch = cur_epoch + FLAGS.evalStep
            if cur_epoch > FLAGS.n_epochs:
                break
    exit()
