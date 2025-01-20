# python main.py --config_name progressive_my --mode POSE --device cuda:0 --data 5gan-5dif-2real
python main.py --config_name progressive_my --mode POSE --device cuda:3 --data 5gan-7dif-2mergereal 
# python main.py --config_name progressive --mode POSE --device cuda:2 --data split1
# python -m debugpy --listen 5678 --wait-for-client 

# python test.py --model_path /home/lihao/python_proj/AIGC_2025/others_work/POSE/dataset/5gan-5dif-2real/5gan-5dif-2real_train/models/5gan-5dif-2real_val/progressive_my/0116/POSE_seed0/model_29_test91.62_acc_AUC_82.38_OSCR_78.98183381410267_acc0.43044736842105263.pth \
# --device cuda:3 --data 5gan-5dif-2real