from argparse import ArgumentParser
import numpy as np
import torch
from models import spectral_attention
import random
from torch.backends import cudnn
from scipy import io
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import seaborn as sns
import cv2

random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
cudnn.deterministic = True
from operator import truediv

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

device = "cuda:0"

from train_IP_speed_normal import HSI_SLIC_data

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum =np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def model_eval(net, data_loader,w,h,palette,epoch=0):
    net.to(device)
    net.eval()

    predicts = np.zeros((0))
    targets = np.zeros((0))
    xs = np.zeros((0))
    ys = np.zeros((0))
    indicesX = []
    indicesY = []
    for iter_, (data, target, x, y) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        # data, target, x, y = data.to(device), target.to(device), x.to(device), y.to(device)
        try:
            output, _ = net(data)
        except:
            output = net(data)
        _, predict = torch.max(output.data, 1)
        predict = predict.cpu().numpy()
        target = target.cpu().numpy()
        # x = x.cpu().numpy
        # y = y.cpu().numpy
        predicts = np.append(predicts, predict)
        targets = np.append(targets, target)
        # xs = np.append(xs, x)
        # ys = np.append(ys, y)
        indicesX.extend(x.cpu().numpy().tolist())
        indicesY.extend(y.cpu().numpy().tolist())


    predicts = predicts.tolist()
    targets = targets.tolist()
    
    test_inference_color_map = np.zeros((w, h, 3),dtype=np.uint8)
    for i,(x,y) in enumerate(zip(indicesX,indicesY)):
        # print(i,x,y)
        label = int(predicts[i])+1
        test_inference_color_map[x,y,:] = palette[label]
    cv2.imwrite('./test_inference_result.png',test_inference_color_map)
    
    kappa = cohen_kappa_score(predicts, targets)
    print("kappa: ",kappa)
    con_matrics = confusion_matrix(predicts, targets)
    acc = accuracy_score(predicts, targets)
    print("OA: ",acc)
    
    each_acc, aa = AA_andEachClassAccuracy(con_matrics)
    print("AA: ",aa)
    print("each_acc: ",each_acc)
    print(classification_report(predicts, targets, digits=4))
    print(con_matrics)

    test_acc_list.append(acc)

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    # return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
    return total_pos_train, total_pos_test

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default = './checkpoints/4981_0.907145.pth')
    parser.add_argument('--img_dir', default= './result.png')
    args = parser.parse_args()
    net = spectral_attention(patch_size = 3, # HSI channel merge size 
                sample_size =  15 * 15, # sample area size = h*w (default shape : square  ) 
                head_num=16, # head_num in transformer backbone
                num_classes = 16,
                pool_method = 'mean')
    net.load_state_dict(torch.load(args.model))


    gt = io.loadmat('./Datasets/IndianPines/IndianPines_GT.mat')['groundTruth']

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", np.max(gt))):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    w, h = gt.shape

    gt_color_map = np.zeros((w, h, 3),dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            label = gt[x,y]
            gt_color_map[x,y,:] = palette[label]
    cv2.imwrite('./gt_result.png',gt_color_map)

    

    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices]
    test_gt = np.zeros_like(gt)

    # train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=0.1, random_state = 0,stratify = y)
    # test_indices = list(zip(*test_indices))


    temp_data = io.loadmat('./Datasets/IndianPine.mat')
    TR = temp_data['TR']
    TE = temp_data['TE']
    temp_label = TR + TE
    temp_num_classes = np.max(TR)
    total_pos_train, total_pos_test = chooose_train_and_test_point(TR, TE, temp_label, temp_num_classes)
    test_indices = list(zip(*total_pos_test))
    

  

    test_gt[test_indices] = gt[test_indices]

    img = io.loadmat('./Datasets/IndianPines/IndianPines.mat')['imageCube']# Load image to numpy.ndarray
    # ['indian_pines_corrected']
    img = (img - np.min(img))/(np.max(img)-np.min(img))     # Normalization

    
    sr = io.loadmat('Datasets/IndianPines/segmentation_results_200.mat')['segmentation_results']
    test_dataset = HSI_SLIC_data(img, test_gt, sr)                                     
    # test_dataset = HSI_SLIC_data(img, test_gt, sr)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100,# batch_size=512,
                                          shuffle=True)
    # TODO: 可视化结果

    test_gt_color_map = np.zeros((w, h, 3),dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            label = test_gt[x,y]
            test_gt_color_map[x,y,:] = palette[label]
    cv2.imwrite('./test_gt_result.png',test_gt_color_map)

    
    test_acc_list = []
    model_eval(net, test_loader,w,h,palette,epoch=0)
    # print(test_acc_list)
    print("done!")
