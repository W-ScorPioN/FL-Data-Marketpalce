import numpy as np
import logging
import datetime
import torch


# 加载模型权重
def load_model_weights(file_path, device):
    model_state = torch.load(file_path, map_location=device)
    # 将模型权重展开为一维向量
    weights = []
    for key in model_state:
        weights.append(model_state[key].flatten().cpu().numpy())  # 将张量转为CPU以便计算
    return np.concatenate(weights)


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


# 求标准偏差
def get_bias(type, list, args):
    avg_list = sum(list) / len(list)  
    bias_list = 0
   
    for i in list:
        bias_list += (i - avg_list)**2
        
    bias_list = np.sqrt(bias_list / len(list))
   
    logging.info(f'The {type} of the {args.dataset}: {avg_list:.6f} ± {bias_list:.6f}.')



# 保存
def save_info(group_dict, path, args, i):
    f = open(path + args.dataset + '_' + args.model + '_' + str(args.epochs) + '_' + str(args.num_users) + '_' + str(args.frac) + '_' + '_' + args.task_id + str(i) + '.txt', 'w+') 
    f.write(str(group_dict))
    f.close()


# 读取
def read_info(path, args, i):
    f = open(path + args.dataset + '_' + args.model + '_' + str(args.epochs) + '_' + str(args.num_users) + '_' + str(args.frac) + '_' + str(args.num_groups) + '_' + str(args.num_models) + '_' + str(args.alpha) + '_' + args.task_id + str(i) + '.txt', 'r+') 
    line = f.readline()
    res = ast.literal_eval(line)
    f.close()
    return res