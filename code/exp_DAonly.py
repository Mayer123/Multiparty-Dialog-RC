import numpy as np
import random
import tensorflow as tf
import os
from keras import backend as k
from optparse import OptionParser
import time
import sys
import json
import tools
from collections import Counter
from collections import OrderedDict
from nn_models import *
from keras.models import load_model
import logging

def gen_examples(x1, x2, l, y, qmask, dmask, batch_size):
    minibatches = tools.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_qmask = qmask[minibatch]
        mb_dmask = dmask[minibatch]
        all_ex.append((mb_x1, mb_x2, mb_l, mb_y, mb_qmask, mb_dmask))
    return all_ex

def pre_shuffle(x1, x2, l, y, qmask, dmask):
    combine = list(zip(x1, x2, l, y, qmask, dmask))
    np.random.shuffle(combine)
    x1, x2, l, y, qmask, dmask = zip(*combine)
    return list(x1), list(x2), np.array(l), list(y), np.array(qmask), np.array(dmask)

def accuracy_score(y_pred, y_true):
    assert len(y_pred) == len(y_true)

    if len(y_true) == 0:
        return 0.

    correctly = 0

    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correctly += 1

    return float(correctly)

def eval_acc(any_model, all_examples, max_d, max_q, max_s):
    acc = 0
    n_examples = 0
    for x1, x2, l, y, q_mask, d_mask in all_examples:
        newx1 = []
        for i in xrange(len(x1[0])):
            newx1.append(np.array([scene[i] for scene in x1]))
        predictions = any_model.predict_classes(newx1+[np.array(x2)]+[np.array(l)]
            +[np.array(q_mask)]+[np.array(d_mask)], np.array(l))
        
        dev_pred = accuracy_score(predictions, y)
        acc += dev_pred
        n_examples += len(x1)
    return acc * 100.0 / n_examples

def cnn_lstm_DA(args):
    logger_exp.info('-' * 50)
    logger_exp.info('Load data files..')
    # get prune dictionaries 
    redundent_1, redundent_2 = tools.prune_data(args.train_file)
    # load training data
    train_examples, max_d, max_q, max_s = tools.load_jsondata(args.train_file, redundent_1, redundent_2, args.stopwords)
    # load development data
    dev_examples, a, b, c = tools.load_jsondata(args.dev_file, redundent_1, redundent_2, args.stopwords)
   
    num_train = len(train_examples[0])
    num_dev = len(dev_examples[0])
    logger_exp.info('-' * 50)
    logger_exp.info('Build dictionary..')
    word_dict = tools.build_dict(train_examples[0], train_examples[1])
    # entity dictionary for entire dataset
    entity_markers = list(set([w for w in word_dict.keys()
                              if w.startswith('@ent')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    logger_exp.info('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)

    logger_exp.info('-' * 50)
    # Load embedding file
    embeddings = tools.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
   
    (vocab_size, args.embedding_size) = embeddings.shape
    logger_exp.info('Building Model..')
    # build model
    if args.model_to_run == 'cnn_lstm_DA':
        cnn_model = CNN_LSTM_DA_Model('CNN_LSTM_DA_Model', num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)
    if args.model_to_run == 'cnn_lstm':
        cnn_model = CNN_LSTM_Model('CNN_LSTM_Model', num_labels, vocab_size, args.embedding_size, max_d, max_q, max_s,
            nb_filters_utterance=args.utterance_filters, nb_filters_query=args.query_filters, learning_rate=args.learning_rate,
            dropout=args.dropout, nb_hidden_unit=args.hidden_size)

    cnn_model.load_embedding(np.array([embeddings]))
    if args.pre_trained is not None:
        cnn_model.load_weights(args.pre_trained)

    logger_exp.info('Done.')

    logger_exp.info('-' * 50)
    logger_exp.info(args)

    logger_exp.info('-' * 50)
    logger_exp.info('Intial test..')
    # vectorize development data
    dev_x1, dev_x2, dev_l, dev_y, dev_qmask, dev_dmask = tools.vectorize(dev_examples, word_dict, entity_dict, max_d, max_q, max_s)
    assert len(dev_x1) == num_dev
    
    all_dev = gen_examples(dev_x1, dev_x2, dev_l, dev_y, dev_qmask, dev_dmask, args.batch_size)
    dev_acc = eval_acc(cnn_model, all_dev, max_d, max_q, max_s)
    logger_exp.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    if args.test_only:
        return
    cnn_model.save_model(args.save_model)

    # Training
    logger_exp.info('-' * 50)
    logger_exp.info('Start training..')

    # vectorize training data
    train_x1, train_x2, train_l, train_y, train_qmask, train_dmask = tools.vectorize(train_examples, word_dict, entity_dict, max_d, max_q, max_s)
    assert len(train_x1) == num_train
    
    train_x1, train_x2, train_l, train_y, train_qmask, train_dmask = pre_shuffle(train_x1, train_x2, train_l, train_y, train_qmask, train_dmask)
    start_time = time.time()
    n_updates = 0
    all_train = gen_examples(train_x1, train_x2, train_l, train_y, train_qmask, train_dmask, args.batch_size)

    for epoch in range(args.nb_epoch):
        np.random.shuffle(all_train)
        
        for idx, (mb_x1, mb_x2, mb_l, mb_y, mb_qmask, mb_dmask) in enumerate(all_train):
            logger_exp.info('#Examples = %d' % (len(mb_x1)))
            # rearrange each batch of dialogs
            newx1 = []
            for i in xrange(len(mb_x1[0])):
                newx1.append(np.array([scene[i] for scene in mb_x1]))   
            
            hist = cnn_model.fit(newx1+[np.array(mb_x2)]+[np.array(mb_l)]
                +[np.array(mb_qmask)]+[np.array(mb_dmask)], np.array(mb_y), 
                batch_size=args.batch_size, verbose=0)
            logger_exp.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, idx, len(all_train), hist.history['loss'][0], time.time() - start_time))
            n_updates += 1
            # evaluate every 100 batches 
            if n_updates % 100 == 0:
                samples = sorted(np.random.choice(num_train, min(num_train, num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            train_l[samples],
                                            [train_y[k] for k in samples],
                                            train_qmask[samples],
                                            train_dmask[samples],
                                            args.batch_size)
                
                logger_exp.info('Train accuracy: %.2f %%' % eval_acc(cnn_model, sample_train, max_d, max_q, max_s))
                dev_acc = eval_acc(cnn_model, all_dev, max_d, max_q, max_s)
                logger_exp.info('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logger_exp.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    cnn_model.save_model(args.save_model)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config=config))
    model_dict = ['cnn_lstm', 'cnn_lstm_DA']
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--logging_to_file',
                      action='store',
                      dest='logging_to_file',
                      default=None,
                      help='logging to a file or stdout')
    parser.add_option('--model',
                      action='store',
                      dest='model_to_run',
                      default=None,
                      help='model to train (available: %s)' % ', '.join(model_dict))
    parser.add_option('--nb_epoch',
                      action='store',
                      dest='nb_epoch',
                      default=100,
                      help='number of epochs to train the model with')
    parser.add_option('--embedding_size',
                      action='store',
                      dest='embedding_size',
                      default=100,
                      help='embedding size of the inputs')
    parser.add_option('--train_file',
                      action='store',
                      dest='train_file',
                      default=None,
                      help='train file')
    parser.add_option('--dev_file',
                      action='store',
                      dest='dev_file',
                      default=None,
                      help='dev file')
    parser.add_option('--save_model',
                      action='store',
                      dest='save_model',
                      default=None,
                      help='model to save')
    parser.add_option('--random_seed',
                      action='store',
                      dest='random_seed',
                      default=1234,
                      help='random seed')
    parser.add_option('--embedding_file',
                      action='store',
                      dest='embedding_file',
                      default=None,
                      help='embedding file')
    parser.add_option('--stopwords',
                      action='store',
                      dest='stopwords',
                      default=None,
                      help='stopwords')
    parser.add_option('--test_only',
                      action='store',
                      dest='test_only',
                      default=False,
                      help='If just to test the model')
    parser.add_option('--pre_trained',
                      action='store',
                      dest='pre_trained',
                      default=None,
                      help='pre-trained model')
    parser.add_option('--batch_size',
                      action='store',
                      dest='batch_size',
                      default=32,
                      help='training and testing batch size')
    parser.add_option('--hidden_size',
                      action='store',
                      dest='hidden_size',
                      default=32,
                      help='hidden size of LSTM')
    parser.add_option('--query_filters',
                      action='store',
                      dest='query_filters',
                      default=50,
                      help='number of filters for query CNN')
    parser.add_option('--utterance_filters',
                      action='store',
                      dest='utterance_filters',
                      default=50,
                      help='number of filters for utterance CNN')
    parser.add_option('--dropout',
                      action='store',
                      dest='dropout',
                      default=0.2,
                      help='dropout rate for LSTM')
    parser.add_option('--learning_rate',
                      action='store',
                      dest='learning_rate',
                      default=0.001,
                      help='learning rate of the model')
    (options, args) = parser.parse_args()

    fixed_seed_num = int(options.random_seed)
    options.nb_epoch = int(options.nb_epoch)
    options.batch_size = int(options.batch_size)
    options.embedding_size = int(options.embedding_size)
    options.hidden_size = int(options.hidden_size)
    options.query_filters = int(options.query_filters)
    options.utterance_filters = int(options.utterance_filters)
    options.dropout = float(options.dropout)
    options.learning_rate = float(options.learning_rate)

    np.random.seed(fixed_seed_num)
    random.seed(fixed_seed_num)
    tf.set_random_seed(fixed_seed_num)
    if options.train_file is None:
        raise ValueError('train_file is not specified.')
    if options.dev_file is None:
        raise ValueError('dev_file is not specified.')
    if options.stopwords is None:
        raise ValueError('stopwords are not specified.')
    if options.embedding_file is not None:
        dim = tools.get_dim(options.embedding_file)
        if (options.embedding_size is not None) and (options.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                                (options.embedding_size, options.embedding_file, dim))
    else:
        raise ValueError('embedding_file is not specified.')

    FORMAT = '[%(levelname)-8s] [%(asctime)s] [%(name)-15s]: %(message)s'
    DATEFORMAT = '%Y-%m-%d %H:%M:%S'

    if options.logging_to_file:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT,
                            filename=options.logging_to_file)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT)

    logger_exp = logging.getLogger('experiments')
   
    if not options.logging_to_file:
        logger_exp.info('logging to stdout')

    if options.model_to_run not in model_dict:
        raise Exception('model `%s` not implemented' % options.model_to_run)

    cnn_lstm_DA(options)

	