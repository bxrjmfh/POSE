import os
import numpy as np
import importlib

import torch
from torch.utils.data import DataLoader

from utils.common import plot_hist_seaborn, plot_ROC_curve, read_annotations, load_config
from utils.evaluation import evaluate_multiclass, metric_ood, compute_oscr
from data.dataset import ImageDataset
from models.models import Simple_CNN

def get_feature(model, dataloader, config, device):
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
            prob, feature = model(input_img, data='dct')
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

def calculate_open_set_result(_labels_k, _labels_u, _pred_k, _pred_u, known_feature, unknown_feature, known_classes, unknown_classes, save_dir):
    
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    out_results = metric_ood(x1, x2)['Bas'] # check
    _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels_k)
    unknown_perf = round(out_results['AUROC'], 2)

    print("AUC, OSCR : {:.2f} {:.2f}".format(unknown_perf, _oscr_socre*100))

    # plot confidence histogram and ROC curve
    os.makedirs(save_dir, exist_ok=True)
    plot_hist_seaborn(x1, x2, save_path = os.path.join(save_dir,'hist.png'))
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

device = 'cuda:3'
config = load_config('configs.{}'.format('progressive'))

# load model
model_path = './checkpoints/split1/model.pth'
model = Simple_CNN(class_num=15, out_feature_result=True)
pretrained_dict = torch.load(model_path, map_location='cpu')['state_dict']
model.load_state_dict(pretrained_dict)
model = model.to(device)

test_data_path = '/home/lihao/python_proj/AIGC_2025/others_work/POSE/dataset/split1_test/annotations/split1_test.txt'
close_set = ImageDataset(read_annotations(test_data_path), config, balance=False, test_mode=True)
close_loader = DataLoader(
    dataset=close_set,
    num_workers=config.num_workers,
    batch_size=config.batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=False,
)
known_feature, known_label, known_prob  = get_feature(model, close_loader, config, device)

out_data_path = '/home/lihao/python_proj/AIGC_2025/others_work/POSE/dataset/split1_test/annotations/split1_test_out_arch.txt'
out_set = ImageDataset(read_annotations(out_data_path), config, balance=False, test_mode=True)
out_loader = DataLoader(
    dataset=out_set,
    num_workers=config.num_workers,
    batch_size=config.batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=False,
)
out_feature, out_label, out_prob = get_feature(model, out_loader, config, device)

known_classes = ['real', 'celeba_ProGAN_seed0', 'celeba_StarGAN', 'FFHQ_StyleGAN3_r', 'FFHQ_StyleGAN3_t', 'SNGAN_imagenet', 'SAGAN_imagenet', 'lsun_ProGAN_seed0', 'lsun_MMDGAN', 'FaceSwap', 'FSGAN', 'stylegan_cat', 'stylegan3_cat', 'stylegan_bus', 'progan_bus']
unknown_classes = ['celeba_MMDGAN', 'celeba_AttGAN', 'celeba_SNGAN', 'celeba_InfomaxGAN', \
                   'FFHQ_StyleGAN2', 'CelebA_HQ_ProGAN', 'CelebA_HQ_StyleGAN', \
                   'BigGAN_imagenet', 'S3GAN_imagenet', 'ContraGAN_imagenet', \
                   'lsun_SNGAN', 'lsun_InfomaxGAN', \
                   'Faceshifter', 'wav2lip', \
                   'stylegan2_cat', 'progan_cat', 'mmdgan_cat', 'sngan_cat', \
                   'stylegan2_bus', 'stylegan3_bus', 'mmdgan_bus', 'sngan_bus']
calculate_open_set_result(known_label, out_label, known_prob, out_prob, known_feature, out_feature, known_classes, unknown_classes, './results/unseen_arch')