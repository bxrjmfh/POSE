import os
known_classes= [ "ldm-text2im-large-256",
                "controlnet-canny-sdxl-1.0", "lcm-lora-sdv1-5", "sd-turbo",
                "sdxl-turbo", "styleGAN3", "styleGAN", "SNGAN", "ProGAN", "biggan"]

# unknown_classes= ['real_coco', 'real_lsun', 'real_pose', 
#                     'ADM', 'stable_diffusion_v_1_5', 'glide',
#                     'wukong', 'VQDM', 'Midjourney', 
#                     'MMDGAN', 'SAGAN', 'S3GAN', 
#                     'FaceShifter', 'InfoMax-GAN', 'wav2lip', 
#                     'starGAN', 'FaceSwap', 'styleGAN2', 
#                     'ContraGAN', 'AttGAN', 'FSGAN', 
#                     'VAR']

eval_classes= [ "ldm-text2im-large-256",
                "controlnet-canny-sdxl-1.0", "lcm-lora-sdv1-5", "sd-turbo",
                "sdxl-turbo", "styleGAN3", "styleGAN", "SNGAN", "ProGAN", "biggan",'VAR']

base_dir = './dataset/' # data base
data_base_dir = './dataset/dif-gan/mix_data_new'
exp_name = '5gan-5dif'

unknown_classes = [x for x in eval_classes if x not in known_classes]

os.makedirs(os.path.join(base_dir,exp_name),exist_ok=True)
train_collections = exp_name+'_train'
val_collections = exp_name+'_val'
os.makedirs(os.path.join(base_dir,exp_name,train_collections,'annotations'),exist_ok=True)
os.makedirs(os.path.join(base_dir,exp_name,val_collections,'annotations'),exist_ok=True)

def merge_real(classes,split,data_base_dir):
    d = {'real':[]}
    for k in classes:
        path = os.path.join(data_base_dir,split,k)
        data_li = os.listdir(path)
        if 'real' in k:
            d['real'] += [os.path.join(path,it) for it in data_li]
        else:
            d[k] = [os.path.join(path,it) for it in data_li]
    if d['real'] == []:
        d.pop('real')
    return d


known_data_d = merge_real(known_classes,'train',data_base_dir)
known_test_data_d = merge_real(known_classes,'eval',data_base_dir)
unknown_test_data_d = merge_real(unknown_classes,'eval',data_base_dir)

print(f"known data d: {len(known_data_d)}")
print(f"known test data d: {len(known_test_data_d)}")
print(f"unknown test data d: {len(unknown_test_data_d)}")

import random
# train
num_train = 5000
num_val = 1000
num_test = 1000
train_res = []
val_res = []

def count_real_classes_in_list(li, known_classes):
    """
    该函数用于统计给定列表中属于已知类别 `real_*` 的出现次数。

    :param li: 需要统计的字符串列表
    :param known_classes: 包含类别名称的列表
    :return: 返回一个字典，包含每个 `real_*` 类别及其出现的次数
    """
    cnt = {x: 0 for x in known_classes if 'real' in x}  # 初始化字典，存储每个 'real' 类别的计数
    real_k = [x for x in known_classes if 'real' in x]  # 获取所有 'real' 开头的类别

    # 遍历传入的列表 `li`，检查每个元素是否包含 `real_*` 类别
    for it in li:
        for x in real_k:
            if x in it:
                cnt[x] += 1  # 如果找到匹配的类别，则计数加 1
    print(cnt)

print('---collecting real train ---')
for i,n in enumerate(known_data_d):
    random.seed(0)
    li = random.sample(known_data_d[n],k=num_train+num_val)
    if n == 'real':
        count_real_classes_in_list(li, known_classes)
    print(f"{n}: {len(li)}")
    for it in li[:num_train]:
        train_res.append(f"{it}\t{i}\n")
    for it in li[num_train:num_train+num_val]:
        val_res.append(f"{it}\t{i}\n")
with open(f'/home/lihao/python_proj/AIGC_2025/others_work/POSE/dataset/{exp_name}/{train_collections}/annotations/{train_collections}.txt','w' ) as f:
    f.writelines(train_res)
with open(f'/home/lihao/python_proj/AIGC_2025/others_work/POSE/dataset/{exp_name}/{val_collections}/annotations/{val_collections}.txt','w' ) as f:
    f.writelines(val_res)
    
# eval
if 'real' in unknown_test_data_d:
    unknown_real = unknown_test_data_d.pop('real')
    known_test_data_d['real'] += unknown_real

print('---collecting test train ---')
tn_res = []
tu_res = []
for i,n in enumerate(known_test_data_d):
    random.seed(0)
    li = random.sample(known_test_data_d[n],k=num_test)
    if n == 'real':
        count_real_classes_in_list(li, eval_classes)
    print(f"{n}: {len(li)}")
    for it in li:
        tn_res.append(f"{it}\t{i}\n")
        
print('---collecting test unseen ---')
for i,n in enumerate(unknown_test_data_d):
    random.seed(0)
    li = random.sample(unknown_test_data_d[n],k=num_test)
    if n == 'real':
        count_real_classes_in_list(li, unknown_classes)
    print(f"{n}: {len(li)}")
    for it in li:
        tu_res.append(f"{it}\t{i}\n")
        

with open(f'./dataset/{exp_name}/test_know_data.txt','w' ) as f:
    f.writelines(tn_res)
with open(f'./dataset/{exp_name}/test_unknow_data.txt','w' ) as f:
    f.writelines(tu_res)

import yaml
# 将数据写入 YAML 文件
data={
    exp_name:{
        'data_path': os.path.join(base_dir,exp_name),
        'known_classes': [k for k in known_data_d],
        'unknown_classes': [k for k in unknown_test_data_d],
        'train_collection':train_collections,
        'val_collection':val_collections,
        'test_data_path': f'./dataset/{exp_name}/test_know_data.txt',
        'out_data_path': f'./dataset/{exp_name}/test_unknow_data.txt'
    }
}
with open('output.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=None,indent=4)