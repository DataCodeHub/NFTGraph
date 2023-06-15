import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from pygod.metrics import eval_precision_at_k,eval_recall_at_k,eval_roc_auc,eval_average_precision
from pygod.models import DOMINANT,AnomalyDAE,DONE,AdONE,GAAN,GCNAE,CONAD,MLPAE
import numpy as np
import random
import os
import datetime
from tqdm import tqdm


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_log(*x, sep="\n", end="\n", mode="a"):
    print(*x, sep=sep, end=end)
    with open(log_file, mode=mode) as f:
        print(*x, sep=sep, end=end, file=f)


model_name = 'AdONE'
log_file = f"./results/{model_name}_result.txt"
runs = 5
seed = 123

f = open(log_file, "w")
f.write(f"{model_name}:\n")
f.close()

ylabel = pd.read_csv('./ylabel-tinynft.csv')
cat_dict = torch.load('/NFTGraph/ogb_graph/example_direct_use/submission_ogbn_tinynft_nodetype/meta_dict.pt')
cat_dict['dir_path'] = '/NFTGraph/ogb_graph/example_direct_use/submission_ogbn_tinynft_nodetype'
dataset = PygNodePropPredDataset(name = 'ogbn-tinynft',root = cat_dict['dir_path'] ,meta_dict=cat_dict)
data = dataset[0]
data.y = torch.tensor(ylabel.values)
print_log(f"Data: {data}\n\n\n")

all_pred_labels = []
all_outlier_scores = []
all_auc_scores = []
all_precision_at_ks = []
all_recall_at_ks = []
all_average_precisions = []
all_tdeltas = []
all_memos = []

for run in range(runs):
    set_random_seed(seed+run)
    stime = datetime.datetime.now()
    memo = torch.cuda.memory_allocated()
    model = AdONE(batch_size=512,gpu=0,epoch=100)
    model.fit(data)

    labels = model.predict(data)
    print_log(f"Labels: {labels}")
    print_log(f"labels.sum(): {labels.sum()}")
    all_pred_labels.append(labels)

    outlier_scores = model.decision_function(data)
    print_log(f"Raw scores: {outlier_scores}")
    all_outlier_scores.append(outlier_scores)

    precision_at_k = eval_precision_at_k(data.y.numpy(), outlier_scores,100)
    print_log(f"precision_at_k: {precision_at_k}")
    all_precision_at_ks.append(precision_at_k)

    recall_at_k = eval_recall_at_k(data.y.numpy(), outlier_scores,100)
    print_log(f"recall_at_k: {recall_at_k}")
    all_recall_at_ks.append(recall_at_k)

    auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
    print_log(f"AUC Score: {auc_score}")
    all_auc_scores.append(auc_score)

    average_precision = eval_average_precision(data.y.numpy(), outlier_scores)
    print_log(f"average_precision: {average_precision}")

    all_average_precisions.append(average_precision)
    tend = datetime.datetime.now()
    etime = datetime.datetime.now()
    print_log(f"end run [{run}], {etime}, spent time:{etime-stime}")
    all_tdeltas.append(etime-stime)
    all_memos.append(memo)
    print_log(f"\n\n\n-------------Finished {run} -------------------------------------\n\n\n")



precision_at_k_mean = np.mean(all_precision_at_ks)
precision_at_k_std = np.std(all_precision_at_ks)

recall_at_k_mean = np.mean(all_recall_at_ks)
recall_at_k_std = np.std(all_recall_at_ks)

auc_score_mean = np.mean(all_auc_scores)
auc_score_std = np.std(all_auc_scores)

average_precision_mean = np.mean(all_average_precisions)
average_precision_std = np.std(all_average_precisions)

print_log(f"\n\nall_pred_labels:{str(all_pred_labels)}")
print_log(f"all_outlier_scores:{str(all_outlier_scores)}")
print_log(f"all_precision_at_ks:{str(all_precision_at_ks)}")
print_log(f"all_recall_at_ks:{str(all_recall_at_ks)}")
print_log(f"all_average_precisions:{str(all_average_precisions)}")
print_log(f"all_tdeltas:",{str(all_tdeltas)})

print_log("\n\nResults:")
print_log(f"precision_at_k: {precision_at_k_mean} ± {precision_at_k_std}")
print_log(f"recall_at_k: {recall_at_k_mean} ± {recall_at_k_std}")
print_log(f"auc_score: {auc_score_mean} ± {auc_score_std}")
print_log(f"average_precision: {average_precision_mean} ± {average_precision_std}")

