'''
Load the output.txt file and compute the matrics
'''

import numpy as np
import operator
import random
from collections import defaultdict
import metrics


test_out_filename = "output_test.txt"
print("*"*20 + test_out_filename + "*"*20 + "\n")

with open(test_out_filename, 'r') as f:

    results = defaultdict(list)
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        us_id = line[0]
        r_id = line[1]
        prob_score = float(line[2])
        label = float(line[4])
        results[us_id].append((r_id, label, prob_score))

    accu, precision, recall, f1, loss = metrics.classification_metrics(results)
    print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))
    total_valid_query = metrics.get_num_valid_query(results)
    mvp = metrics.mean_average_precision(results)
    mrr = metrics.mean_reciprocal_rank(results)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tNum_query: {}'.format(
        mvp, mrr, total_valid_query))
    top_1_precision = metrics.top_k_precision(results, k=1)
    top_2_precision = metrics.top_k_precision(results, k=2)
    top_5_precision = metrics.top_k_precision(results, k=5)
    top_10_precision = metrics.top_k_precision(results, k=10)
    print('Recall@1: {}\tRecall@2: {}\tRecall@5: {}\tRecall@10: {}\n'.format(
        top_1_precision, top_2_precision, top_5_precision, top_10_precision))

