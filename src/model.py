import numpy as np
import os, time
import tensorflow as tf
import sys
import tensorflow.contrib.layers as layers
import utils
import safe_borderline

'''
This code implements the NeuCrowd framework and its variants

''' 

data_name = 'hotel'
SYM = False
dimension = 300

groupSize = 6e4

# Use hard sampler from the 2nd epoch.
warm_up = 0
warm_up_sampler = 1

MODEL_NAME = 'NeuCrowd_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}_dropout_{}' + \
             '_hard_{}_sim_{}_ON_{}'

SAMPLER_MODEL_NAME = 'SAMPLER_NeuCrowd_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}_dropout_{}' + \
             '_hard_{}_sim_{}_ON_{}'

LOG_NAME = "../train_logs/{}"
config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True

# compute cosine distance between two vectors


def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
    return cos_similarity


def euc_sim(a, b):
    euc_simimarity = 1 - tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
    return euc_simimarity



class RLL(object):
    def __init__(self, dimension, l1_n, l2_n, gamma, sim_type):

        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
        self.sim_type = sim_type

    def loadData(self, raw_train_data, raw_validation_data):
        """
        Data loading.
        Refer to ../data/hotel for detailed data format.
        
        """

        self.raw_train_data = raw_train_data
        self.raw_validation_data = raw_validation_data

    def feedBatch(self, groups, weights, batchSize, lr_rate, dropout_rate, reg_scale, is_training):
        batchIndex = np.random.randint(low=0, high=groups[0].shape[0], size=batchSize)
        batchGroups = [groups[i][batchIndex] for i in range(len(groups))]
        batchWeights = [weights[i][batchIndex] for i in range(len(weights))]
        batchData = {
                            self.is_training: is_training,
                            self.lr_rate: lr_rate,
                            self.dropout_rate: dropout_rate,
                            self.reg_scale: reg_scale,

                            self.query: batchGroups[0],
                            self.posDoc: batchGroups[1],
                            self.negDoc0: batchGroups[2],
                            self.negDoc1: batchGroups[3],
                            self.negDoc2: batchGroups[4],

                            self.queryDocW: batchWeights[0].reshape(-1, ),
                            self.posDocW: batchWeights[1].reshape(-1, ),
                            self.negDoc0W: batchWeights[2].reshape(-1, ),
                            self.negDoc1W: batchWeights[3].reshape(-1, ),
                            self.negDoc2W: batchWeights[4].reshape(-1,)

                    }
        raw_batch_data = {
                            'is_training': is_training,
                            'lr_rate': lr_rate,
                            'dropout_rate': dropout_rate,
                            'reg_scale': reg_scale,

                            'query': batchGroups[0],
                            'posDoc': batchGroups[1],
                            'negDoc0': batchGroups[2],
                            'negDoc1': batchGroups[3],
                            'negDoc2': batchGroups[4],

                            'queryDocW': batchWeights[0].reshape(-1, ),
                            'posDocW': batchWeights[1].reshape(-1, ),
                            'negDoc0W': batchWeights[2].reshape(-1, ),
                            'negDoc1W': batchWeights[3].reshape(-1, ),
                            'negDoc2W': batchWeights[4].reshape(-1,)
                    }
        return batchData, raw_batch_data

    def single_net(self, input_layer, reuse=False):

        with tf.name_scope('fc_l1'):
            output = tf.contrib.layers.fully_connected(input_layer, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                                 activation_fn = tf.nn.sigmoid, scope='fc_l1',
                                                                 biases_initializer=tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        with tf.name_scope('fc_l2'):
            output = tf.contrib.layers.fully_connected(output, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l2',
                                                            biases_initializer = tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        return output

    def buildRLL(self):
        tf.reset_default_graph()

        self.is_training = tf.placeholder_with_default(False, shape=(), name='isTraining')

        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension], name='queryInput')
        self.posDoc = tf.placeholder(tf.float32, shape=[None, self.dimension], name='posDocInput')
        self.negDoc0 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc0Input')
        self.negDoc1 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc1Input')
        self.negDoc2 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc2Input')

        self.queryDocW = tf.placeholder(tf.float32, shape=[None], name='queryDocWeight')
        self.posDocW = tf.placeholder(tf.float32, shape=[None], name='posDocWeight')
        self.negDoc0W = tf.placeholder(tf.float32, shape=[None], name='negDoc0Weight')
        self.negDoc1W = tf.placeholder(tf.float32, shape=[None], name='negDoc1Weight')
        self.negDoc2W = tf.placeholder(tf.float32, shape=[None], name='negDoc2Weight')

        self.lr_rate = tf.placeholder(tf.float32, shape=(), name='learningRate')
        self.reg_scale = tf.placeholder(tf.float32, shape=(), name='penaltyScale')
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropoutKeepRate')

        if self.sim_type == 'cos':
            sim = cos_sim
        elif self.sim_type == 'euc':
            sim = euc_sim

        # Shared weights.
        outputQuery = self.single_net(input_layer=self.query)
        outputPosDoc = self.single_net(input_layer=self.posDoc, reuse=True)
        outputNegDoc0 = self.single_net(input_layer=self.negDoc0, reuse=True)
        outputNegDoc1 = self.single_net(input_layer=self.negDoc1, reuse=True)
        outputNegDoc2 = self.single_net(input_layer=self.negDoc2, reuse=True)

        #########

        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)


            # assurance weighted anchor
            outputQuery = tf.reduce_mean(tf.multiply(self.queryDocW[:, tf.newaxis], outputQuery), axis=0, keep_dims=True)

            self.query_pos_sim = sim(outputQuery, outputPosDoc)
            self.query_doc0_sim = sim(outputQuery, outputNegDoc0)
            self.query_doc1_sim = sim(outputQuery, outputNegDoc1)
            self.query_doc2_sim = sim(outputQuery, outputNegDoc2)

            nominator = tf.multiply(self.posDocW, tf.exp(tf.multiply(self.gamma, self.query_pos_sim)))
            doc0_similarity = tf.multiply(self.negDoc0W, tf.exp(tf.multiply(self.gamma, self.query_doc0_sim)))
            doc1_similarity = tf.multiply(self.negDoc1W, tf.exp(tf.multiply(self.gamma, self.query_doc1_sim)))
            doc2_similarity = tf.multiply(self.negDoc2W, tf.exp(tf.multiply(self.gamma, self.query_doc2_sim)))
            self.prob = prob = tf.add(nominator, tf.constant(1e-7))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator, tf.constant(1e-7))
            log_prob = tf.log(prob)
            self.loss = -tf.reduce_sum(log_prob) + reg_loss

            tf.summary.histogram('nominator', nominator)
            tf.summary.histogram('doc0_similarity', doc0_similarity)
            tf.summary.histogram('doc1_similarity', doc1_similarity)
            tf.summary.histogram('doc2_similarity', doc2_similarity)
            tf.summary.histogram('prob', prob)
            tf.summary.scalar('prob_loss', -tf.reduce_sum(log_prob))
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge_all()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr_rate).minimize(self.loss)


    def train(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }

        _, batch_loss = sess.run([self.optimizer, self.loss], feed_dict=batchData)
        return batch_loss

    def validate(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }

        batch_loss = sess.run(self.loss, feed_dict=batchData)
        return batch_loss

    def embedding_forward_loss(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }
        prob_as_difficulty = np.array(sess.run(self.prob, feed_dict=batchData))

        return prob_as_difficulty 

    def get_embedding(self, sess, X_input):

        graph = tf.get_default_graph()
        X = tf.placeholder(tf.float32, shape=[None, dimension], name='input')
        w1 = graph.get_tensor_by_name('fc_l1/weights:0')
        b1 = graph.get_tensor_by_name('fc_l1/biases:0')
        w2 = graph.get_tensor_by_name('fc_l2/weights:0')
        b2 = graph.get_tensor_by_name('fc_l2/biases:0')
        embd = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(X, w1) + b1), w2) + b2)
        feed = {X: X_input}
        output = sess.run(embd, feed_dict=feed)
        return np.array(output)


"""
Sampler net
"""


class RLLSampler(object):
    def __init__(self, dimension, l1_n, l2_n, gamma, hard_rate):

        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
        self.hard_rate = hard_rate
        self.use_hard_rate = 1

    def single_net(self, input_layer, reuse=False):

        with tf.name_scope('fc_l1_s'):
            output = tf.contrib.layers.fully_connected(input_layer, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                                 activation_fn = tf.nn.sigmoid, scope='fc_l1_s',
                                                                 biases_initializer=tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        with tf.name_scope('fc_l2_s'):
            output = tf.contrib.layers.fully_connected(output, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l2_s',
                                                            biases_initializer = tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        return output

    def buildSampler(self):

        self.is_training = tf.placeholder_with_default(False, shape=(), name='isTraining_s')

        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension], name='queryInput_s')
        self.posDoc = tf.placeholder(tf.float32, shape=[None, self.dimension], name='posDocInput_s')
        self.negDoc0 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc0Input_s')
        self.negDoc1 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc1Input_s')
        self.negDoc2 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc2Input_s')

        self.queryDocW = tf.placeholder(tf.float32, shape=[None], name='queryDocWeight_s')
        self.posDocW = tf.placeholder(tf.float32, shape=[None], name='posDocWeight_s')
        self.negDoc0W = tf.placeholder(tf.float32, shape=[None], name='negDoc0Weight_s')
        self.negDoc1W = tf.placeholder(tf.float32, shape=[None], name='negDoc1Weight_s')
        self.negDoc2W = tf.placeholder(tf.float32, shape=[None], name='negDoc2Weight_s')

        self.lr_rate = tf.placeholder(tf.float32, shape=(), name='learningRate_s')
        self.reg_scale = tf.placeholder(tf.float32, shape=(), name='penaltyScale_s')
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropoutKeepRate_s')

        self.loss_input = tf.placeholder(tf.float32, shape=[None], name='loss_input_s')

        outputQuery = self.single_net(input_layer=self.query)
        outputPosDoc = self.single_net(input_layer=self.posDoc, reuse=True)
        outputNegDoc0 = self.single_net(input_layer=self.negDoc0, reuse=True)
        outputNegDoc1 = self.single_net(input_layer=self.negDoc1, reuse=True)
        outputNegDoc2 = self.single_net(input_layer=self.negDoc2, reuse=True)

        #########

        with tf.name_scope('loss_s'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_s')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_s')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)

            concat = tf.concat(
                [outputQuery, outputPosDoc, outputNegDoc0, outputNegDoc1, outputNegDoc2], axis=1)


            self.prob = prob = tf.contrib.layers.fully_connected(concat, 1,
                                                              activation_fn=tf.nn.sigmoid, scope='prob_s',
                                                              biases_initializer=tf.random_normal_initializer())

            # add regularization
            self.loss = tf.reduce_sum(tf.square(self.prob - self.loss_input), keep_dims=True) + reg_loss

            tf.summary.histogram('prob_s', prob)
            tf.summary.scalar('reg_loss_s', reg_loss)
            tf.summary.scalar('loss_s', self.loss)

        self.summaries = tf.summary.merge_all()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr_rate).minimize(self.loss)

    def train(self, sess, batchData, sampler_input_loss):

        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }

        sampler_input_loss = {self.loss_input: sampler_input_loss}


        _, batch_loss = sess.run([self.optimizer, self.loss], feed_dict={**batchData, **sampler_input_loss})
        return batch_loss

    def sampler_forward_loss(self, sess, batchData):

        prob_as_difficulty = np.array(sess.run(self.prob, feed_dict=batchData))

        return prob_as_difficulty # the smaller, the harder.

    def hard_feedBatch(self, sess, groups, weights, batchSize,
                       lr_rate, dropout_rate, reg_scale, is_training):
        """


        """
        candidates_size = int(batchSize * self.hard_rate)

        batchIndex = np.random.randint(low=0, high=groups[0].shape[0], size=candidates_size)
        batchGroups = [groups[i][batchIndex] for i in range(len(groups))]
        batchWeights = [weights[i][batchIndex] for i in range(len(weights))]

        batchData = {
            self.is_training: is_training,
            self.lr_rate: lr_rate,
            self.dropout_rate: dropout_rate,
            self.reg_scale: reg_scale,

            self.query: batchGroups[0],
            self.posDoc: batchGroups[1],
            self.negDoc0: batchGroups[2],
            self.negDoc1: batchGroups[3],
            self.negDoc2: batchGroups[4],

            self.queryDocW: batchWeights[0].reshape(-1, ),
            self.posDocW: batchWeights[1].reshape(-1, ),
            self.negDoc0W: batchWeights[2].reshape(-1, ),
            self.negDoc1W: batchWeights[3].reshape(-1, ),
            self.negDoc2W: batchWeights[4].reshape(-1, )
        }

        prob = self.sampler_forward_loss(sess, batchData)
        
        # choose instances with the smallest batchSize probs.
        prob = np.array([i[0] for i in prob])
        hard_index = prob.argsort()[::1][0:batchSize]

        batchData = {
            self.is_training: is_training,
            self.lr_rate: lr_rate,
            self.dropout_rate: dropout_rate,
            self.reg_scale: reg_scale,

            self.query: batchGroups[0][hard_index],
            self.posDoc: batchGroups[1][hard_index],
            self.negDoc0: batchGroups[2][hard_index],
            self.negDoc1: batchGroups[3][hard_index],
            self.negDoc2: batchGroups[4][hard_index],

            self.queryDocW: batchWeights[0][hard_index].reshape(-1, ),
            self.posDocW: batchWeights[1][hard_index].reshape(-1, ),
            self.negDoc0W: batchWeights[2][hard_index].reshape(-1, ),
            self.negDoc1W: batchWeights[3][hard_index].reshape(-1, ),
            self.negDoc2W: batchWeights[4][hard_index].reshape(-1, )

        }
        raw_batch_data = {
                            'is_training': is_training,
                            'lr_rate': lr_rate,
                            'dropout_rate': dropout_rate,
                            'reg_scale': reg_scale,

                            'query': batchGroups[0][hard_index],
                            'posDoc': batchGroups[1][hard_index],
                            'negDoc0': batchGroups[2][hard_index],
                            'negDoc1': batchGroups[3][hard_index],
                            'negDoc2': batchGroups[4][hard_index],

                            'queryDocW': batchWeights[0][hard_index].reshape(-1, ),
                            'posDocW': batchWeights[1][hard_index].reshape(-1, ),
                            'negDoc0W': batchWeights[2][hard_index].reshape(-1, ),
                            'negDoc1W': batchWeights[3][hard_index].reshape(-1, ),
                            'negDoc2W': batchWeights[4][hard_index].reshape(-1,)

                    }
        # print('\n\n')
        # print(hard_index)
        # print(prob)
        # print(np.squeeze(batchGroups[2][hard_index], axis=1)[1: 4])
        # print(batchWeights[2][hard_index].reshape(-1, )[1: 4])
        return batchData, raw_batch_data


class RLLWORKER:
    # put them together

    def __init__(self, dimension, l1_n, l2_n, gamma, hard_rate, sim_type):
        self.RLL = RLL(dimension, l1_n, l2_n, gamma, sim_type)
        self.RLLSampler = RLLSampler(dimension, l1_n, l2_n, gamma, hard_rate)
        self.rll_path = '../model_neucrowd/rll'
        self.sampler_path = '../model_neucrowd/rll_sampler'

    def build(self, raw_train_data, raw_validation_data):
        self.RLL.buildRLL()
        self.RLLSampler.buildSampler()
        self.RLL.loadData(raw_train_data=raw_train_data, raw_validation_data=raw_validation_data)

    # 
    def train(self, batchSize, lr_rate, reg_scale, dropout_rate, max_iter):

        # 
        earlyStopCount = 0
        saver = tf.train.Saver(max_to_keep=1)

        # 
        model_name = MODEL_NAME.format(self.RLL.l1_n, self.RLL.l2_n, lr_rate, reg_scale, batchSize, dropout_rate,
                                       self.RLLSampler.hard_rate, self.RLL.sim_type, data_name)
        print('training model {}'.format(model_name))

        sampler_model_name = SAMPLER_MODEL_NAME.format(self.RLL.l1_n, self.RLL.l2_n, lr_rate, reg_scale, batchSize,
                                                       dropout_rate, self.RLLSampler.hard_rate,
                                                       self.RLL.sim_type, data_name)


        currentModelPath = os.path.join(self.rll_path, model_name)
        if(not os.path.exists(currentModelPath)):
            os.makedirs(currentModelPath)

        currentSamplerPath = os.path.join(self.sampler_path, sampler_model_name)
        if(not os.path.exists(currentSamplerPath)):
            os.makedirs(currentSamplerPath)


        with tf.Session(config=config) as sess:

            tf.global_variables_initializer().run()
            batch_writer = tf.summary.FileWriter(LOG_NAME.format(model_name))
            batch_sampler_writer = tf.summary.FileWriter(LOG_NAME.format(sampler_model_name))

            """
            Start training.
            """

            best_val_loss = sys.maxsize

            for epoch in range(max_iter):

                # prepare to filter unsafe instances.
                X_input = self.RLL.raw_train_data[:, 2:]
                X_label = self.RLL.raw_train_data[:, 0]

                # confidence here stands for assurance.
                data_with_confidence = utils.inferWeight(self.RLL.raw_train_data)
                X_confidence = data_with_confidence[:, 1]
                safe_data_idx, unsafe_data_idx, _ = safe_borderline.safe_borderline(self.RLL.get_embedding(sess, X_input),
                                                                X_label, X_confidence)
                if epoch > warm_up:
                    try:
                        # you can set a rate to let part of unsafe samples in.
                        safe_rate = 50 #  when chosen into groups, safe: unsafe = 50: 1

                        unsafe_data_idx = np.array(unsafe_data_idx)
                        hard_length = unsafe_data_idx.shape[0]
                        np.random.shuffle(unsafe_data_idx)
                        sampled_hard = unsafe_data_idx[:hard_length // safe_rate]

                        groupsTr, weightsTr = utils.prepareInput(self.RLL.raw_train_data[safe_data_idx + sampled_hard.tolist(), :], groupSize=int(groupSize))
                        print('use safety')

                    except:
                        groupsTr, weightsTr = utils.prepareInput(self.RLL.raw_train_data, groupSize=int(groupSize))
                        print('not use safety')


                else:
                    groupsTr, weightsTr = utils.prepareInput(self.RLL.raw_train_data, groupSize=int(groupSize))

                groupsVal, weightsVal = utils.prepareInput(self.RLL.raw_validation_data, groupSize=int(groupSize))

                train_confidence_record_before_sample.append([(epoch, np.mean(np.array(weightsTr)), np.std(np.array(weightsTr)))])

                if epoch == 0:
                    train_size = groupsTr[0].shape[0]
                    print('training group size is {}'.format(train_size))
                    val_size = groupsVal[0].shape[0]
                    print('validation group size is {}'.format(val_size))

                num_batch = train_size // batchSize


                # batch
                total_loss = 0
                total_sampler_loss = 0
                for batch in range(num_batch):

                    if epoch < warm_up_sampler:
                        _, batchData = self.RLL.feedBatch(groupsTr, weightsTr, batchSize, lr_rate,
                                                       dropout_rate, reg_scale, is_training=True)

                    else:
                        self.RLLSampler.use_hard_rate = self.RLLSampler.hard_rate
                        _, batchData = self.RLLSampler.hard_feedBatch(sess, groupsTr, weightsTr, batchSize, lr_rate,
                                                                   dropout_rate, reg_scale, is_training=True)

                    sampler_input_loss = self.RLL.embedding_forward_loss(sess, batchData)

                    if batch == 0:
                        train_difficulty_record_after_sample.append((epoch, [batchData, sampler_input_loss]))

                    sampler_output_loss = self.RLLSampler.train(sess, batchData, sampler_input_loss)
                    total_sampler_loss += sampler_output_loss

                    batch_loss = self.RLL.train(sess, batchData)
                    total_loss += batch_loss

                print("Epoch {} train loss {}".format(epoch, total_loss / train_size))
                # print("Epoch {} sampler_net loss {}".format(epoch, total_sampler_loss / train_size))
                train_loss_record.append((epoch, total_loss / train_size))
                # sampler_loss_record.append((epoch, total_sampler_loss / train_size))

                if epoch % 3 == 0:
                    _, valData = self.RLL.feedBatch(groupsVal, weightsVal, groupsVal[0].shape[0], lr_rate, dropout_rate,
                                                 reg_scale, is_training=False)

                    valLoss = self.RLL.validate(sess, valData)
                    print('*' * 66)
                    print("Epoch {} validation loss {}".format(epoch, valLoss / val_size))
                    print('\n')
                    val_loss_list_record.append((epoch, valLoss / val_size))
                    if best_val_loss > valLoss:
                        best_val_loss = valLoss
                        print('best val loss is', best_val_loss)
                        earlyStopCount = 0
                        saver.save(sess, os.path.join(currentModelPath, model_name + '.ckpt'))
                    elif epoch > 10:
                        earlyStopCount += 1
                        # print(earlyStopCount)

                if (earlyStopCount >= 15):
                    print('Early stop at epoch {}!'.format(epoch))
                    break
