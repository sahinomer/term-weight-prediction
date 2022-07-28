import tensorflow as tf
import tensorflow_hub as hub
from official import nlp
import official.nlp.optimization

from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.layers.pooling import GlobalPooling1D

CLS = 101
SEP = 102
MASK = 103


class GlobalMaskedMaxPooling1D(GlobalPooling1D):

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == 'channels_last' else 2

        if mask is not None:
            mask = math_ops.cast(mask, backend.floatx())
            mask = array_ops.expand_dims(
                mask, 2 if self.data_format == 'channels_last' else 1)

            # mask[cond] = -inf
            mask = backend.log(mask)
            inputs = inputs + mask

        return backend.max(inputs, axis=steps_axis)


class BERTModel:
    def __init__(self, num_train_steps, warmup_steps, bert_model='bert_en_uncased_L-12_H-768_A-12/3'):
        self.model = None
        self.vocabulary = None
        self.lower_case = None

        self.bert_url = f'https://tfhub.dev/tensorflow/{bert_model}'

        self.optimizer = nlp.optimization.create_optimizer(
            2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    def build_model(self, model='sequence'):
        input_word_ids = Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(None,), dtype=tf.int32, name="input_mask")
        input_type_ids = Input(shape=(None,), dtype=tf.int32, name="input_type_ids")
        bert_inputs = {'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}

        bert_encoder = hub.KerasLayer(self.bert_url, trainable=True, name='bert_encoder')
        bert_output = bert_encoder(bert_inputs)

        if model.startswith('sequence'):
            sequence_output = bert_output['sequence_output']

            if 'cls_token' in model:
                term_mask = tf.logical_and(tf.equal(input_word_ids, CLS), tf.equal(input_type_ids, 0))
                query_mask = tf.logical_and(tf.equal(input_word_ids, CLS), tf.equal(input_type_ids, 1))

                term_feature = GlobalMaskedMaxPooling1D(name='term_cls_token')(sequence_output, mask=term_mask)
                query_feature = GlobalMaskedMaxPooling1D(name='query_cls_token')(sequence_output, mask=query_mask)

            else:
                # Term/Query mask ######################################################################################
                token_mask = tf.logical_and(tf.not_equal(input_word_ids, CLS, name='except_cls'),
                                            tf.not_equal(input_word_ids, SEP, name='except_sep'),
                                            name='except_cls_sep')

                token_mask = tf.logical_and(tf.equal(input_mask, 1, name='attention_mask'),
                                            token_mask, name='token_attention')

                term_mask = tf.logical_and(tf.equal(input_type_ids, 0, name='term_part'),
                                           token_mask, name='except_cls_sep')

                query_mask = tf.logical_and(tf.equal(input_type_ids, 1, name='query_part'),
                                            token_mask, name='except_cls_sep')

                term_feature = GlobalMaskedMaxPooling1D(name='max_term_pooling')(sequence_output, mask=term_mask)
                query_feature = GlobalMaskedMaxPooling1D(name='max_query_pooling')(sequence_output, mask=query_mask)

                ########################################################################################################

            term_query_abs_diff = tf.abs(tf.subtract(term_feature, query_feature, name='subtract_term-query'),
                                         name='absolute_difference')
            term_query_multiply = tf.multiply(term_feature, query_feature, name='multiply_term-query')

            feature_vector = tf.concat([term_feature, query_feature, term_query_abs_diff, term_query_multiply],
                                       axis=-1, name='concat_features')

        else:  # BERT pooled output
            feature_vector = bert_output['pooled_output']

        dropout_layer = Dropout(rate=0.2, name='dropout')(feature_vector)
        regression = Dense(1, name='output')(dropout_layer)

        self.model = Model(inputs=bert_inputs, outputs=regression)

        self.model.compile(optimizer=self.optimizer, loss='mse')

        self.model.summary()

        self.vocabulary = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
        self.lower_case = bert_encoder.resolved_object.do_lower_case.numpy()

    def train(self, generator, epochs=2, checkpoint_path='checkpoints/checkpoint'):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=512 * generator.batch_size)

        self.model.fit(generator, epochs=epochs, verbose=1, callbacks=[cp_callback])
        self.model.save_weights(checkpoint_path)

    def evaluate(self, generator):
        return self.model.evaluate(generator, verbose=1)

    def predict(self, x):
        return self.model(x).numpy()

    def load_from_checkpoint(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)
