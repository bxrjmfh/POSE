import os
import numpy as np
import importlib
import argparse

import torch
from torch.utils.data import DataLoader

from utils.common import plot_hist, plot_ROC_curve, read_annotations, load_config
from utils.evaluation import evaluate_multiclass, metric_ood, compute_oscr, metric_cluster
from utils.config import Config
from models.models import Simple_CNN
from data.dataset import ImageDataset

from sklearn.metrics import confusion_matrix

def get_feature(model, dataloader, config, device, input_data):
    model.eval()
    Loss = importlib.import_module('loss.' + config.loss)
    criterion = getattr(Loss, config.loss)(config).to(device)
    with torch.no_grad():
        features = []
        labels =[]
        probs=[]
        for i, batch in enumerate(dataloader):
            input_img_batch, label_batch, _ = batch 
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(device)
            label = label_batch.reshape((-1)).to(device)
            prob, feature = model(input_img, data=input_data)
            prob, _ = criterion(feature, prob)

            if i == 0:
                probs = prob
                gt_labels = label
                features = feature
            else:
                probs = torch.cat([probs, prob], dim=0)
                gt_labels = torch.cat([gt_labels, label])
                features=torch.cat(([features, feature]))

    features = features.cpu().numpy()
    probs = probs.cpu().numpy()
    labels = gt_labels.cpu().numpy()
        
    return features, labels, probs


# def calculate_open_set_result(_labels_k, _pred_k, _pred_u, save_dir):
    
#     x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
#     out_results = metric_ood(x1, x2)['Bas'] # check
#     _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels_k)
#     unknown_perf = round(out_results['AUROC'], 2)

#     print("AUC, OSCR : {:.2f} {:.2f}".format(unknown_perf, _oscr_socre*100))

#     # plot confidence histogram and ROC curve
#     plot_hist(x1, x2, save_path = os.path.join(save_dir,'hist.png'))
#     plot_ROC_curve(out_results, save_path = os.path.join(save_dir,'roc.png'))

def calculate_open_set_result(_labels_k, _labels_u, _pred_k, _pred_u, known_classes, unknown_classes, save_dir):
    
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    out_results = metric_ood(x1, x2)['Bas'] # check
    _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels_k)
    unknown_perf = round(out_results['AUROC'], 2)

    print("AUC, OSCR : {:.2f} {:.2f}".format(unknown_perf, _oscr_socre*100))

    # plot confidence histogram and ROC curve
    os.makedirs(save_dir, exist_ok=True)
    plot_hist(x1, x2, save_path = os.path.join(save_dir,'hist.png'))
    plot_ROC_curve(out_results, save_path = os.path.join(save_dir,'roc.png'))
    
    # print detailed results for each unknown class
    for i, label_u in enumerate(set(_labels_u)):
        pred_u = _pred_u[_labels_u==label_u]
        x1, x2 = np.max(_pred_k, axis=1), np.max(pred_u, axis=1)
        pred = np.argmax(pred_u, axis=1)
        pred_labels = list(set(pred))
        pred_nums = [np.sum(pred==p) for p in pred_labels]
        result = metric_ood(x1, x2, verbose=False)['Bas']
        print("{}\t \t mostly pred class: {}\t \t average score: {}\t AUROC (%): {:.2f}".format(unknown_classes[i], 
                                                                                 known_classes[pred_labels[np.argmax(pred_nums)]],
                                                                                 np.mean(x2), result['AUROC']))


def calculate_closed_set_result(known_prob, known_label, classes_names=None):
    pred_labels = np.argmax(known_prob, axis=1)
    results = evaluate_multiclass(known_label, pred_labels)
    CM = confusion_matrix(known_label, pred_labels)
    perf = round(results['acc'], 4)*100
    
    if classes_names:
        # assert len(classes_names) == len(old_classes_gt) + len(new_classes_gt)
        for i,name in enumerate(classes_names):
            every_class_acc = CM[i,i] *1.0/ CM[i].sum()
            print("{} acc: {:.2f}".format(classes_names[i],every_class_acc))
    print('closed_set accuracy', perf)
    print(CM)

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--model_path', type=str, default='', help='pretrain_model_path', required=True)
    parser.add_argument('--device', type=str, default='cuda:0', help='device', required=True)
    parser.add_argument('--input_data', type=str, default='dct')
    parser.add_argument('--data', type=str, default='split1')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_config('configs.{}'.format('progressive'))
    data_lists = Config(config_filepath='./configs/data_list.yaml')
    data_list = data_lists[args.data]
    known_classes = data_list['known_classes']
    unknown_classes = data_list['unknown_classes']
    test_data_path, out_data_path = data_list['test_data_path'], data_list['out_data_path']

    close_set = ImageDataset(read_annotations(test_data_path), config, balance=False, test_mode=True)
    close_loader = DataLoader(
        dataset=close_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    out_set = ImageDataset(read_annotations(out_data_path), config, balance=False, test_mode=True)
    out_loader = DataLoader(
        dataset=out_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )

    model = Simple_CNN(class_num=len(data_list['known_classes']), out_feature_result=True)
    pretrained_dict = torch.load(args.model_path, map_location='cpu')['state_dict']
    model.load_state_dict(pretrained_dict)
    model = model.to(args.device)
    
    model_dir = os.path.split(args.model_path)[0]
    save_dir = os.path.join(model_dir, 'pred')
    os.makedirs(save_dir, exist_ok=True)
    
    known_feature, known_label, known_prob  = get_feature(model, close_loader, config, args.device, args.input_data)
    calculate_closed_set_result(known_prob, known_label, classes_names=known_classes)

    out_feature, out_label, out_prob  = get_feature(model, out_loader, config, args.device, args.input_data)
    calculate_open_set_result(known_label, out_label, known_prob, out_prob, known_classes, unknown_classes, save_dir)

    features = np.concatenate([known_feature, out_feature])
    labels = np.concatenate([known_label,  out_label+len(set(known_label))]) # 存疑
    class_num = len(set(known_label)) + len(set(out_label))
    mask = np.zeros_like(labels, dtype=bool)
    mask[:len(known_label)] = True
    NMI, cluster_acc, purity, gcd_acc_v1, gcd_acc_v2 = metric_cluster(features, class_num, labels, mask, 'kmeans',class_names=known_classes+unknown_classes)
    # print(f"NMI: {NMI}, cluster acc: {cluster_acc}, purity: {purity}")
    # all_acc, old_acc, new_acc = 
    # print(f"all acc: {all_acc}, old acc: {old_acc}, new acc: {new_acc}")
    
if __name__=='__main__':
    main()

