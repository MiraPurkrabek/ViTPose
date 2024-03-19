import os
import json
import numpy as np

from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


ROOT = "/datagrid/personal/purkrmir/data/OOI_eval/coco_cropped_v2/"
# ROOT = "/datagrid/personal/purkrmir/data/OOI_eval/coco_mpii_cropped_v2/"
models = [
    # ("ViTPose_small_coco_256x192", "vanilla", False),
    # ("ViTPose_small_coco_256x192_blackout_unfreeze", "black", False),
    ("ViTPose_small_coco_256x192_full_blackout_finetune", "conf", False),
    ("ViTPose_small_coco_256x192_full_blackout_finetune", "prob", True),
    
]

GT = json.load(open("{:s}/annotations/person_keypoints_val2017.json".format(ROOT), "r"))

gt_vis = []
anns = [(str(ann["image_id"]), ann["keypoints"]) for ann in GT["annotations"]]
anns = sorted(anns, key=lambda x: x[0])
for ann in anns:
    gt_vis.append(ann[1][2::3])
gt_vis = np.array(gt_vis).flatten()
# Ignore keypoints with visibility 0
ignore_mask = gt_vis > 1
gt_vis = gt_vis[ignore_mask]
gt_vis[gt_vis == 3] = 0
# gt_vis[gt_vis == 1] = 0
gt_vis[gt_vis == 2] = 1

print("Before sampling")
print(np.unique(gt_vis, return_counts=True))

# Sample data such that the number of visible keypoints is balanced
sampled_v0_idx = np.where(gt_vis == 0)[0]
np.random.shuffle(sampled_v0_idx)
sampled_v1_idx = np.where(gt_vis == 1)[0]
np.random.shuffle(sampled_v1_idx)

n_idx = min(len(sampled_v0_idx), len(sampled_v1_idx))
sampled_v0_idx = sampled_v0_idx[:n_idx]
sampled_v1_idx = sampled_v1_idx[:n_idx]

gt_vis = np.concatenate([gt_vis[sampled_v0_idx], gt_vis[sampled_v1_idx]])
print("After sampling")
print(np.unique(gt_vis, return_counts=True))

for model in models:

    PRED = json.load(open(os.path.join(
        ROOT,
        "test_visualization",
        model[0],
        "test_results.json",    
    ), "r"))


    pred_vis = []
    if model[2]:
        anns = [(k, v["prob"]) for k, v in PRED.items()]
    else:
        anns = [(k, v["keypoints"][2::3]) for k, v in PRED.items()]
    anns = sorted(anns, key=lambda x: x[0])
    for ann in anns:
        pred_vis.append(ann[1])
    pred_vis = np.array(pred_vis).flatten()
    # if model[2]:
    #     pred_vis = pred_vis /2
    pred_vis = pred_vis[ignore_mask]

    pred_vis = np.concatenate([pred_vis[sampled_v0_idx], pred_vis[sampled_v1_idx]])

    # print("Accuracy", metrics.accuracy_score(gt_vis, pred_vis))


    fpr, tpr, _ = metrics.roc_curve(gt_vis,  pred_vis, )
    auc = metrics.roc_auc_score(gt_vis, pred_vis)
    
    prec, rec, thr = metrics.precision_recall_curve(gt_vis, pred_vis)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    print("{:10s}: best F1 score is {:.3f} at threshold {:.3f}".format(model[1], np.max(f1), thr[np.argmax(f1)]))

    plt.plot(fpr,tpr,label="{:s}, auc={:.3f}".format(model[1], auc), linewidth=2.0)
    
    # plt.plot(rec,prec,label="{:s}, auc={:.2f}".format(model[1], auc), linewidth=1.0)

plt.legend(loc=4)
plt.grid(True)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc.png")