import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import accuracy_score,  roc_auc_score, f1_score
import warnings
import pickle
import pandas as pd

warnings.simplefilter('ignore', category=Warning, lineno=0, append=False)
 
DATA_NAME = sys.argv[1]  # 'hotel'

dimension = 300

np.random.seed(1)

############


def load_data(mode='train'):  # train, test, valid

    data_path = '../data/{}/{}.csv'.format(DATA_NAME, mode)
    #
    data = pd.read_csv(data_path)

    input_y = np.array(data.iloc[:, 0])
    input_X = np.array(data.iloc[:, 2:])

    return input_X, input_y


def reloadGraph(modelPath):
    tf.reset_default_graph()
    sess = tf.Session()
    
    metaFile = modelPath.split('/')[-1]+'.ckpt.meta'
    saver = tf.train.import_meta_graph(os.path.join(modelPath, metaFile))
    saver.restore(sess, tf.train.latest_checkpoint(modelPath))
    graph = tf.get_default_graph()
    return graph, sess


def get_embed_feature(graph, loaded_sess, inputX):
    X = tf.placeholder(tf.float32, shape=[None, dimension], name='input')
    sess = loaded_sess
    #with loaded_sess as sess:
    w1 = graph.get_tensor_by_name('fc_l1/weights:0')
    b1 = graph.get_tensor_by_name('fc_l1/biases:0')
    w2 = graph.get_tensor_by_name('fc_l2/weights:0')
    b2 = graph.get_tensor_by_name('fc_l2/biases:0')
    embd = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(X, w1)+b1), w2)+b2)
    feed = {X: inputX}
    output = sess.run(embd, feed_dict=feed)
    return output


if __name__ == '__main__':

    deep_model_file = 'model_checkpoint_folder'

    train_X, train_y = load_data(mode='train')
    valid_X, valid_y = load_data(mode='valid')

    graph, session = reloadGraph('../model_neucrowd/rll/' + deep_model_file)

    embd = get_embed_feature(graph, session, train_X)
    embd_valid = get_embed_feature(graph, session, valid_X)

    session.close()

    model = LogisticRegression(penalty='l2', C=1, max_iter=400, solver='liblinear', class_weight='balanced')
    model.fit(embd, train_y)

    y_hat = model.predict(embd_valid)
    y_proba = model.predict_proba(embd_valid)

