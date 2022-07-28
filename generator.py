import math

from tensorflow.python.keras.utils.data_utils import Sequence


class TermQueryGenerator(Sequence):
    """
    Data generator that tokenize and encode term/query
    """

    def __init__(self, tokenizer, query_terms, batch_size, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.query_terms = query_terms
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.query_terms['query']) / self.batch_size)

    def __getitem__(self, idx):
        batch_term = self.query_terms['term'][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_query = self.query_terms['query'][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_target = self.query_terms['target'][idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_encode = self.tokenizer(batch_term, batch_query, padding=True, truncation=True,
                                      max_length=self.max_seq_length, return_tensors="tf")

        return {'input_word_ids': batch_encode['input_ids'],
                'input_mask': batch_encode['attention_mask'],
                'input_type_ids': batch_encode['token_type_ids']}, batch_target


########################################################################################################################


def iter_queries(query_list, tokenizer, max_seq_length):
    """
    Iterate query inputs
    """
    for qid, query_dict in query_list:

        # placeholder_pads = ['PAD'] * (batch_size - len(query_dict['term']))

        query_input = tokenizer(query_dict['term'], query_dict['query'],
                                padding=True, truncation=True,
                                max_length=max_seq_length, return_tensors="tf")

        query_input = {'input_word_ids': query_input['input_ids'],
                       'input_mask': query_input['attention_mask'],
                       'input_type_ids': query_input['token_type_ids']}

        yield qid, query_dict['query'][0], query_dict['term'], query_input
