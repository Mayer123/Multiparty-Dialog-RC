from base import BaseModel
from keras.layers import Embedding, Input
from keras.layers.merge import Concatenate, Multiply, dot, Dot
from keras.layers.core import Dense, Lambda, Reshape, Dropout
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.core import Reshape, Permute, Dense, Flatten, Lambda
from keras.activations import softmax
import numpy as np

class CNN_LSTM_UA_DA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        model = self.build_model()
        super(CNN_LSTM_UA_DA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        m_classes = m_classes / masked_sum
        m_classes = K.clip(m_classes, 1e-7, 1.0 - 1e-7)
        return m_classes

    def crossatt(self, x):
        doc, query, doc_mask, q_mask = x[0], x[1], x[2], x[3]
        trans_doc = K.permute_dimensions(doc, (0,2,1))
        match_score = K.tanh(dot([query, trans_doc], (2, 1)))
        query_to_doc_att = K.softmax(K.sum(match_score, axis=1))
        doc_to_query_att = K.softmax(K.sum(match_score, axis=-1))

        alpha = query_to_doc_att*doc_mask
        a_sum = K.sum(alpha, axis=1)
        _a_sum = K.expand_dims(a_sum, -1)
        alpha = alpha/_a_sum

        beta = doc_to_query_att*q_mask
        b_sum = K.sum(beta, axis=1)
        _b_sum = K.expand_dims(b_sum, 1)
        beta = beta/_b_sum

        doc_vector = dot([trans_doc, alpha], (2, 1))
        trans_que = K.permute_dimensions(query, (0,2,1))
        que_vector = dot([trans_que, beta], (2, 1))
        final_hidden = K.concatenate([doc_vector, que_vector])
        return final_hidden

    def build_model(self):

        inputs = []
        # utterances 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # similarity matrices 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token, self.nb_query_token)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)  
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)
        # utternace level attention matrix
        attn = DocAttentionMap((self.nb_utterance_token, self.embedding_size))
        # 3-D embedding for utterances
        embedding_utterances = []
        for i in xrange(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[i]))
            doc_att_map = Reshape((self.nb_utterance_token, self.embedding_size, 1))(attn(inputs[i+self.nb_utterances]))    
            embedding_utterances.append(Concatenate()([embedding_utter, doc_att_map]))
        print embedding_utterances[0]._keras_shape

        # convolution embedding input for query
        conv_embedding_query = Reshape((self.nb_query_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[-4]))
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # convolution output for query
        conv_q = Reshape((self.nb_query_token,self.nb_filters_query))(Convolution2D(self.nb_filters_query, (1, self.embedding_size), activation='relu')(conv_embedding_query))
        # utterance embeddings
        scene = []
        for i in xrange(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(MaxPooling2D(pool_size=(self.nb_utterance_token-j+1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance*4,1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        print scene._keras_shape
        # convolution output of dialog matrix 
        reshape_scene = Reshape((self.nb_utterances, self.nb_filters_utterance*4, 1))(scene)
        single = Convolution2D(self.nb_filters_utterance, (1, self.nb_filters_utterance*4), activation='relu')(reshape_scene)
        single = Reshape((self.nb_utterances, self.nb_filters_utterance))(single)
        
        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        # dialog level attention vector
        att_vector = Lambda(self.crossatt, output_shape=(self.nb_filters_utterance*2,))([single, conv_q, inputs[-1], inputs[-2]]) 

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn, att_vector])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes

class CNN_LSTM_UA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        model = self.build_model()
        super(CNN_LSTM_UA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        m_classes = m_classes / masked_sum
        m_classes = K.clip(m_classes, 1e-7, 1.0 - 1e-7)
        return m_classes

    def build_model(self):

        inputs = []
        # utterances 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # similarity matrices 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token, self.nb_query_token)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)  
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)
        # utternace level attention matrix
        attn = DocAttentionMap((self.nb_utterance_token, self.embedding_size))
        # 3-D embedding for utterances
        embedding_utterances = []
        for i in xrange(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[i]))
            doc_att_map = Reshape((self.nb_utterance_token, self.embedding_size, 1))(attn(inputs[i+self.nb_utterances]))    
            embedding_utterances.append(Concatenate()([embedding_utter, doc_att_map]))
        print embedding_utterances[0]._keras_shape

        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # utterance embeddings
        scene = []
        for i in xrange(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(MaxPooling2D(pool_size=(self.nb_utterance_token-j+1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance*4,1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        print scene._keras_shape
        
        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes

class CNN_LSTM_DA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        model = self.build_model()
        super(CNN_LSTM_DA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        m_classes = m_classes / masked_sum
        m_classes = K.clip(m_classes, 1e-7, 1.0 - 1e-7)
        return m_classes

    def crossatt(self, x):
        doc, query, doc_mask, q_mask = x[0], x[1], x[2], x[3]
        trans_doc = K.permute_dimensions(doc, (0,2,1))
        match_score = K.tanh(dot([query, trans_doc], (2, 1)))
        query_to_doc_att = K.softmax(K.sum(match_score, axis=1))
        doc_to_query_att = K.softmax(K.sum(match_score, axis=-1))

        alpha = query_to_doc_att*doc_mask
        a_sum = K.sum(alpha, axis=1)
        _a_sum = K.expand_dims(a_sum, -1)
        alpha = alpha/_a_sum

        beta = doc_to_query_att*q_mask
        b_sum = K.sum(beta, axis=1)
        _b_sum = K.expand_dims(b_sum, 1)
        beta = beta/_b_sum

        doc_vector = dot([trans_doc, alpha], (2, 1))
        trans_que = K.permute_dimensions(query, (0,2,1))
        que_vector = dot([trans_que, beta], (2, 1))
        final_hidden = K.concatenate([doc_vector, que_vector])
        return final_hidden

    def build_model(self):

        inputs = []
        # utterances 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)  
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)

        # 2-D embedding for utterances
        embedding_utterances = []
        for i in xrange(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[i]))
            embedding_utterances.append(embedding_utter)
        print embedding_utterances[0]._keras_shape

        # convolution embedding input for query
        conv_embedding_query = Reshape((self.nb_query_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[-4]))
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # convolution output for query
        conv_q = Reshape((self.nb_query_token,self.nb_filters_query))(Convolution2D(self.nb_filters_query, (1, self.embedding_size), activation='relu')(conv_embedding_query))
        # utterance embeddings
        scene = []
        for i in xrange(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(MaxPooling2D(pool_size=(self.nb_utterance_token-j+1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance*4,1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        print scene._keras_shape
        # convolution output of dialog matrix 
        reshape_scene = Reshape((self.nb_utterances, self.nb_filters_utterance*4, 1))(scene)
        single = Convolution2D(self.nb_filters_utterance, (1, self.nb_filters_utterance*4), activation='relu')(reshape_scene)
        single = Reshape((self.nb_utterances, self.nb_filters_utterance))(single)
        
        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        # dialog level attention vector
        att_vector = Lambda(self.crossatt, output_shape=(self.nb_filters_utterance*2,))([single, conv_q, inputs[-1], inputs[-2]]) 

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn, att_vector])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes

class CNN_LSTM_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False
        
        model = self.build_model()
        super(CNN_LSTM_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        m_classes = m_classes / masked_sum
        m_classes = K.clip(m_classes, 1e-7, 1.0 - 1e-7)
        return m_classes

    def build_model(self):

        inputs = []
        # utterances 
        for i in xrange(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)  
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)

        # 2-D embedding for utterances
        embedding_utterances = []
        for i in xrange(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(self.embedding_layer_utterance(inputs[i]))
            embedding_utterances.append(embedding_utter)
        print embedding_utterances[0]._keras_shape

        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # utterance embeddings
        scene = []
        for i in xrange(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(MaxPooling2D(pool_size=(self.nb_utterance_token-j+1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance*4,1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        print scene._keras_shape
       
        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        
        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes

class DocAttentionMap(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.U = None
        super(DocAttentionMap, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(DocAttentionMap, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
        
    def build(self, input_shape):
        #print input_shape
        self.U = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim[1]),
                                      initializer='uniform',
                                      trainable=True)
        super(DocAttentionMap, self).build(input_shape)

    def call(self, x, **kwargs):
        #print 'x (Q): %s' % str(x._keras_shape)
        #print 'U    : %s' % str(self.U._keras_shape)
        xU = K.tanh(K.dot(x, self.U))
        return xU

    def compute_output_shape(self, input_shape):
        return None, self.output_dim[0], self.output_dim[1]
