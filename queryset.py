import re
import json
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import shuffle

stop_words = set(stopwords.words('english'))


class QuerySet:
    def __init__(self, max_term_number=None, use_mask=False, query_prefix=None):

        self.max_term_number = max_term_number
        self.use_mask = use_mask
        self.query_prefix = query_prefix

    @staticmethod
    def mask_term(query, term):
        for token in term.split(' '):
            query = re.sub(r'\b{token}\b'.format(token=token), '[MASK]', query, flags=re.IGNORECASE)
        return query

    def get_train_set(self, path, target_weight, samples=None):

        term_list = list()
        query_list = list()
        target_list = list()

        with open(path, mode='r', encoding='utf8') as train_queries:
            for query in train_queries:
                query = json.loads(query)
                for term, weight in query[target_weight].items():

                    if self.max_term_number and len(term.split(' ')) > self.max_term_number:
                        continue

                    query_text = query['query']

                    if self.use_mask:
                        query_text = self.mask_term(query=query_text, term=term)
                    if self.query_prefix:
                        query_text = self.query_prefix + query_text

                    term_list.append(term)
                    query_list.append(query_text)
                    target_list.append(float(weight))

        target_list = np.asarray(target_list)

        if samples and samples < 1:
            samples = int(samples * len(target_list))

        train_term, train_query, train_target = shuffle(term_list, query_list, target_list,
                                                        n_samples=samples, random_state=42)

        train_dict = {'term': train_term, 'query': train_query, 'target': train_target}
        return train_dict

    def get_evaluation_set(self, path, target_weight='term_recall'):
        query_list = list()

        with open(path, mode='r', encoding='utf8') as dev_queries:
            for query in dev_queries:
                query = json.loads(query)

                terms = list(query[target_weight].keys())
                queries = [query['query']] * len(terms)

                if self.use_mask:
                    queries = [self.mask_term(query_text, term) for query_text, term in zip(queries, terms)]
                if self.query_prefix:
                    queries = [self.query_prefix + query_text for query_text in queries]

                query_list.append((query['qid'], {'term': terms, 'query': queries}))

        return query_list
