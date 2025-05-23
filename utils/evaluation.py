import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment as linear_assignment


def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recalls = recall_score(gt_labels, pred_labels, average = None)  # 每一类recall返回
    return {'recalls':recalls,'recall':recall,'f1':f1,'acc':acc}

def get_curve_online(known, novel, stypes = ['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes = ['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes) 
    # tnr_at_tpr95 就是在已知类别分类正确率在 95 以上的时候，未知类别分类为真的比例 tnr=1-fpr
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC' # TPR 和 fpr 曲线的面积
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        roc_auc = 100.*(-np.trapz(1.-fpr, tpr))
        results[stype][mtype] = roc_auc
        results[stype]['tpr']=tpr
        results[stype]['fpr']=fpr

        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC' # for all theta s.t. tpr + tnr == max
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')
    
    return results

def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1) # pred_k和pred_u预测概率的最大值
    pred = np.argmax(pred_k, axis=1) #known样本的预测值
    correct = (pred == labels) 
    m_x1 = np.zeros(len(x1)) 
    m_x1[pred == labels] = 1
    
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0) #known样本的预测目标值
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0) #unkown的预测目标
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict) #unknown+known样本数量

    # Cutoffs are of prediction values
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort() #从小到大排序
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1): # k是阈值，概率大于k位置的样本判断为know,否则为unknown
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR


def metric_cluster(X_selected, n_clusters, y, mask=None, cluster_method='kmeans', class_names=None):
    """
    This function calculates ARI, ACC and NMI of clustering results
    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    if cluster_method == 'kmeans':
        # cluster_alg = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
        #              tol=0.0001, precompute_distances=True, verbose=0,
        #              random_state=None, copy_x=True, n_jobs=1)
        cluster_alg = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, verbose=0,
                random_state=None, copy_x=True)
    elif cluster_method == 'minibatch_kmeans':
        cluster_alg = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=2048)
    elif cluster_method == 'dbscan':
        cluster_alg = DBSCAN(eps=3, min_samples=2)
    else:
        raise ValueError('select kmeans or dbscan for cluster')

    cluster_alg.fit(X_selected)
    y_predict = cluster_alg.labels_

    # # from openworld-gan, same as above
    nmi, purity, ari = cluster_stats(y_predict, y)
    gcd_acc_v1 = split_cluster_acc_v1(y_predict,y,mask)
    gcd_acc_v2 = split_cluster_acc_v2(y_predict,y,mask,classes_names=class_names)
    return nmi, purity, ari, gcd_acc_v1, gcd_acc_v2


def cluster_stats(predicted, targets, save_path=None):
    n_clusters = np.unique(predicted).size
    n_classes  = np.unique(targets).size
    num = np.zeros([n_clusters,n_classes])
    unique_targets = np.unique(targets)
    for i,p in enumerate(np.unique(predicted)):
        class_labels = targets[predicted==p]
        num[i,:] = np.sum(class_labels[:,np.newaxis]==unique_targets[np.newaxis,:],axis=0)
    sum_clusters = np.sum(num,axis=1)
    purity = np.max(num,axis=1)/(sum_clusters+(sum_clusters==0).astype(sum_clusters.dtype))
    indices = np.argsort(-purity)

    if save_path is not None:
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(purity[indices],color='red')
        ax1.set_xlabel('Cluster index')
        ax1.set_ylabel('Purity')
        ax2 = ax1.twinx()
        ax2.plot(sum_clusters[indices])
        ax2.set_ylabel('Cluster size')
        plt.legend(('Purity','Cluster size'))
        plt.show()
        plt.title('Cluster size and purity of discovered clusters')
        plt.savefig(save_path)
    print('Data points {} Clusters {}'.format(np.sum(sum_clusters).astype(np.int64), n_clusters))
    print('Average purity: {:.4f} '.format(np.sum(purity*sum_clusters)/np.sum(sum_clusters))+\
          'NMI: {:.4f} '.format(normalized_mutual_info_score(targets, predicted))+\
          'ARI: {:.4f} '.format(adjusted_rand_score(targets, predicted)))
    avg_purity = np.sum(purity*sum_clusters)/np.sum(sum_clusters) 
    nmi = normalized_mutual_info_score(targets, predicted) 
    ari = adjusted_rand_score(targets, predicted) 
    return nmi, avg_purity, ari

def compute_gcd_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = compute_gcd_acc(y_true[mask], y_pred[mask])
    new_acc = compute_gcd_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc
    print(f"accv1: total acc: {total_acc:.2f}, old acc: {old_acc:.2f}, new acc: {new_acc:.2f}")
    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask, classes_names=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1 # debug check
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind} # map the real to perd label 
    if classes_names:
        # assert len(classes_names) == len(old_classes_gt) + len(new_classes_gt)
        for k in ind_map:
            map_k = ind_map[k]
            every_class_acc = w[map_k,k] *1.0/ w[:,k].sum()
            print("{} acc: {:.2f}".format(classes_names[k],every_class_acc))
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances
    print(f"accv2: total acc: {total_acc:.2f}, old acc: {old_acc:.2f}, new acc: {new_acc:.2f}")
    return total_acc, old_acc, new_acc