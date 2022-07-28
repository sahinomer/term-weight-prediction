from argparse import ArgumentParser

import json
import time

from generator import TermQueryGenerator, iter_queries
from model import BERTModel
import transformers
from queryset import QuerySet
import numpy as np


def run(args):
    query_set = QuerySet(max_term_number=args.max_term_number, use_mask=args.use_mask, query_prefix=args.query_prefix)
    train_set = query_set.get_train_set(args.train_queries, target_weight=args.target_weight, samples=None)

    train_data_size = len(train_set['term'])
    num_train_steps = int(train_data_size / args.batch_size) * args.epochs
    warmup_steps = int(args.epochs * train_data_size * 0.1 / args.batch_size)

    print('%d train steps, %d warmup steps' % (num_train_steps, warmup_steps))

    # Model
    model = BERTModel(num_train_steps, warmup_steps, bert_model=args.bert_model)
    model.build_model(model=args.model_architecture)  # 'pooled' or 'sequence'

    # Tokenizer
    tokenizer = transformers.BertTokenizer(vocab_file=model.vocabulary, do_lower_case=model.lower_case)

    train_generator = TermQueryGenerator(tokenizer=tokenizer, query_terms=train_set,
                                         batch_size=args.batch_size, max_seq_length=args.max_seq_length)

    # Training
    if args.load_model:
        model.load_from_checkpoint(args.checkpoint)
    else:
        model.train(train_generator, epochs=args.epochs)

    # Prediction
    query_list = query_set.get_evaluation_set(args.dev_queries)
    with open(args.output, mode='w', encoding='utf8') as query_file:
        count = 0
        elapsed_time = np.zeros(len(query_list))
        init_time = time.time()

        for qid, query, term_list, query_input in iter_queries(query_list, tokenizer, args.max_seq_length):

            score = model.predict(query_input)

            term_dict = dict()
            for term, weight in zip(term_list, score):
                term_dict[term] = float(weight[0])

            json.dump({'qid': qid, 'query': query, 'term_weight': term_dict}, fp=query_file, ensure_ascii=False)
            query_file.write('\n')  # new line

            end_time = time.time()
            elapsed_time[count] = end_time - init_time
            init_time = end_time

            count += 1
            if count % 100 == 0:
                avg_elapsed_time = elapsed_time[:count].mean()
                print('\r%d queries weighted, %f second/query, %f query/second'
                      % (count, avg_elapsed_time, 1 / avg_elapsed_time), end='')

        avg_elapsed_time = elapsed_time[:count].mean()
        print('\r%d queries weighted, %f second/query, %f query/second'
              % (count, avg_elapsed_time, 1 / avg_elapsed_time))


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Train BERT-based model that predicts query term weights for targeted term weights.')

    parser.add_argument('--train_queries', type=str, help='Path of training queries')
    parser.add_argument('--dev_queries', type=str, help='Path of evaluation/development/test queries')
    parser.add_argument('--output', type=str, help='Path of prediction output')

    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')

    parser.add_argument('--max_term_number', default=1, type=int, help='Max number of term in a phrase')
    parser.add_argument('--use_mask', default=False, type=bool, help='Mask term in the query')
    parser.add_argument('--query_prefix', default=None, type=str, help='Use query prefix')
    parser.add_argument('--target_weight', default='term_weight', type=str, help='Target term weighting')

    parser.add_argument('--max_seq_length', default=128, type=int, help='Maximum sequence length')
    parser.add_argument('--bert_model', default='bert_en_uncased_L-12_H-768_A-12/3',
                        type=str, help='Base BERT model in TF-Hub')
    parser.add_argument('--model_architecture', default='pooled',
                        type=str, help='Model architecture: "sequence" or "pooled"')

    parser.add_argument('--load_model', action='store_true', help='Load model from checkpoint')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint', type=str, help='Model checkpoint path')

    run(parser.parse_args())
