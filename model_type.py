import tensorflow as tf
import numpy as np
import math
class tNaLP(object):
    """
    The proposed NaLP model
    """

    def tFCN(self, o_i, o_j, scope='tFCN', reuse=True):
        """
        type-FCN: Obtain the compatibility feature vector of the type vectors
        """
        with tf.variable_scope(scope, reuse=reuse) as scope:
            g_1 = tf.contrib.layers.fully_connected(tf.concat([o_i, o_j], axis=1), self.n_tFCN, activation_fn=tf.nn.relu)
            return g_1

    def concat_role_val(self, i, picture, tcompatibility_list):
        """
        Concatenate the embeddings of the roles and the corresponding values to construct a "picture"
        """
        role_embed = tf.reshape(tf.nn.embedding_lookup(self.role_embeddings, self.input_x[:, i]), [-1, self.embedding_dim])
        val_embed = tf.reshape(tf.nn.embedding_lookup(self.value_embeddings, self.input_x[:, i+1]), [-1, self.embedding_dim])
        concat_embed = tf.concat([role_embed, val_embed], -1)
        picture = tf.concat([picture, tf.reshape(concat_embed, [-1, 1, concat_embed.get_shape().as_list()[-1]])], 1)

        role_type_embed = tf.reshape(tf.nn.embedding_lookup(self.role_type_embeddings, self.input_x[:, i]), [-1, self.type_embedding_dim])
        val_type_embed = tf.reshape(tf.nn.embedding_lookup(self.value_type_embeddings, self.input_x[:, i+1]), [-1, self.type_embedding_dim])
        t_i_j = self.tFCN(role_type_embed, val_type_embed, reuse=True)
        t_i_j = tf.reshape(t_i_j, [1, tcompatibility_list.get_shape().as_list()[1], tcompatibility_list.get_shape().as_list()[2]])
        tcompatibility_list = tf.concat([tcompatibility_list, t_i_j], 0)
        i = i + 2
        return i, picture, tcompatibility_list
    
    def gFCN(self, o_i, o_j, scope='gFCN', reuse=True):
        """
        g-FCN: Obtain the relatedness feature vector of the role-value pairs o_i and o_j
        """
        with tf.variable_scope(scope, reuse=reuse) as scope:
            g_1 = tf.contrib.layers.fully_connected(tf.concat([o_i, o_j], axis=1), self.n_gFCN, activation_fn=tf.nn.relu)
            return g_1
    
    def process_two(self, i, j, arelatedness):
        """
        Obtain the relatedness feature vector of the i-th and j-th role-value pairs via g-FCN
        """
        o_i = tf.reshape(self.M_Rel[:, i, :], [-1, self.n_filters])
        o_j = tf.reshape(self.M_Rel[:, j, :], [-1, self.n_filters])
        g_i_j = tf.cond(tf.equal(tf.add(i, j), 0), 
                        lambda:self.gFCN(o_i, o_j, reuse=False), 
                        lambda:self.gFCN(o_i, o_j, reuse=True))
        g_i_j = tf.reshape(g_i_j, [1, arelatedness.get_shape().as_list()[1], arelatedness.get_shape().as_list()[2]])
        arelatedness = tf.cond(tf.equal(j, 0), 
                        lambda:g_i_j, 
                        lambda:tf.concat([arelatedness, g_i_j], 0))
        j = j + 1
        return i, j, arelatedness
        
    def process_n(self, i, relatedness_list):
        """
        Obtain the relatedness feature vectors of the i-th role-value pair and all the role-value pairs
        """
        j = tf.constant(0, dtype=tf.int32)
        arelatedness = tf.zeros([1, self.batch_size, self.n_gFCN], dtype=tf.float32)
        _, j, arelatedness = tf.while_loop(cond=lambda i, j, arelatedness:tf.less(j, self.arity), body=self.process_two, loop_vars=[i, j, arelatedness], shape_invariants=[i.get_shape(), j.get_shape(), tf.TensorShape([None, arelatedness.shape[1], arelatedness.shape[2]])])
        relatedness_list = tf.cond(tf.equal(i, 0), 
                        lambda:arelatedness, 
                        lambda:tf.concat([relatedness_list, arelatedness], 0)) 
        i = i + 1 
        return i, relatedness_list
    
    def __init__(self, n_values, n_roles, embedding_dim, type_embedding_dim, n_filters, n_gFCN=1000, n_tFCN=100, batch_size=128, is_trainable=True):
        # input_x: The input facts; input_y: The label of the input fact; arity: The arity of the input facts
        self.input_x = tf.placeholder(tf.int32, [batch_size, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.arity = tf.placeholder(tf.int32, name="arity")
        
        self.embedding_dim = embedding_dim
        self.type_embedding_dim = type_embedding_dim
        self.n_filters = n_filters
        self.n_gFCN = n_gFCN
        self.n_tFCN = n_tFCN
        self.batch_size = batch_size

        # -------- Embedding, obtaining "picture" and type compatibility feature vector --------
        with tf.name_scope("embeddings-picture"):
            bound = math.sqrt(1.0/embedding_dim)
            self.value_embeddings = tf.Variable(tf.random_uniform([n_values, embedding_dim], minval=-bound, maxval=bound), name="value_embeddings")
            self.role_embeddings = tf.Variable(tf.random_uniform([n_roles, embedding_dim], minval=-bound, maxval=bound), name="role_embeddings")

            bound1 = math.sqrt(1.0/type_embedding_dim)
            self.value_type_embeddings = tf.Variable(
                    tf.random_uniform([n_values, type_embedding_dim], minval=-bound1, maxval=bound1), name="value_type_embeddings")
            self.role_type_embeddings = tf.Variable(
                    tf.random_uniform([n_roles, type_embedding_dim], minval=-bound1, maxval=bound1), name="role_type_embeddings")
            
            i = tf.constant(2, dtype=tf.int32)
            n = 2*self.arity
            role_embed = tf.reshape(tf.nn.embedding_lookup(self.role_embeddings, self.input_x[:, 0]), [-1, embedding_dim])
            val_embed = tf.reshape(tf.nn.embedding_lookup(self.value_embeddings, self.input_x[:, 1]), [-1, embedding_dim])
            concat_embed = tf.concat([role_embed, val_embed], -1)
            picture = tf.reshape(concat_embed, [-1, 1, concat_embed.get_shape().as_list()[-1]])

            role_type_embed = tf.reshape(tf.nn.embedding_lookup(self.role_type_embeddings, self.input_x[:, 0]), [-1, self.type_embedding_dim])
            val_type_embed = tf.reshape(tf.nn.embedding_lookup(self.value_type_embeddings, self.input_x[:, 1]), [-1, self.type_embedding_dim])
            t_i_j = self.tFCN(role_type_embed, val_type_embed, reuse=False)
            tcompatibility_list = tf.reshape(t_i_j, [1, t_i_j.get_shape().as_list()[0], t_i_j.get_shape().as_list()[1]])

            _, picture, tcompatibility_list = tf.while_loop(cond=lambda i, picture, tcompatibility_list:tf.less(i, n), 
                    body=self.concat_role_val, loop_vars=[i, picture, tcompatibility_list], 
                    shape_invariants=[i.get_shape(), tf.TensorShape([picture.shape[0], None, picture.shape[2]]), tf.TensorShape([None, tcompatibility_list.shape[1], tcompatibility_list.shape[2]]) ])
            self.picture = picture
            self.picture_expanded = tf.expand_dims(self.picture, -1)
            self.tcompatibility_res = tf.reduce_min(tcompatibility_list, axis=0)

        # -------- Convolution and relatedness evaluation --------
        filter_height = 1
        filter_size = self.picture.get_shape().as_list()[-1]
        with tf.name_scope("convolute-relatedness"):
            # Convolution
            filter_shape = [filter_height, filter_size, 1, n_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="fW1")
            b = tf.Variable(tf.constant(0.0, shape=[n_filters]), name="b1")
            conv = tf.nn.conv2d(
                self.picture_expanded,
                W,
                strides=[1, 1, embedding_dim, 1],
                padding="VALID",
                name="conv")
            # Batch normalization
            conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, is_training=is_trainable)
            # Apply nonlinearity
            conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            conv = tf.squeeze(conv, squeeze_dims=[2])
            self.M_Rel = conv
            
            # Relatedness evaluation
            i = tf.constant(0, dtype=tf.int32)
            relatedness_list = tf.zeros([1, batch_size, n_gFCN], dtype=tf.float32)
            _, relatedness_list = tf.while_loop(cond=lambda i, relatedness_list:tf.less(i, self.arity), body=self.process_n, loop_vars=[i, relatedness_list], shape_invariants=[i.get_shape(), tf.TensorShape([None, relatedness_list.shape[1], relatedness_list.shape[2]])])
            self.relatedness_list = relatedness_list
            self.relatedness_res = tf.reduce_min(self.relatedness_list, axis=0)

        # -------- Final (unnormalized) scores and predictions --------
        with tf.name_scope("output"):
            self.scores = tf.contrib.layers.fully_connected(self.relatedness_res, self.input_y.get_shape()[1].value, activation_fn=None)
            self.tscores = tf.contrib.layers.fully_connected(self.tcompatibility_res, self.input_y.get_shape()[1].value, activation_fn=None)
            #self.predictions = tf.multiply( tf.nn.sigmoid(self.scores), tf.nn.sigmoid(self.tscores) )
            self.predictions = tf.minimum(tf.nn.sigmoid(self.scores), tf.nn.sigmoid(self.tscores))
        
        # -------- Calculate mean cross-entropy loss --------
        with tf.name_scope("loss"):
            self.loss = -tf.reduce_mean(self.input_y * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)) + (1 - self.input_y) * tf.log(tf.clip_by_value(1 - self.predictions, 1e-10, 1.0)))  # input_y: 1 for positive facts and 0 for negative ones

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
