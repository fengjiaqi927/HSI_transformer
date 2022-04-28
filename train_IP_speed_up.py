from scipy import io
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import os
from sklearn.metrics import accuracy_score
import random
from torch.backends import cudnn
from models import *


device = torch.device('cuda:{}'.format(1))
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
cudnn.deterministic = True

datasets = {
        'PaviaC': {
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'img': 'IndianPines.mat',
            'gt': 'IndianPines_GT.mat',
            'sr':'IP_segment_results_100'
            #'sr':'segmentation_results'
            }}



label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                "Corn", "Grass-pasture", "Grass-trees",
                "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                "Stone-Steel-Towers"]



# data split like SpectralFormer
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

class HSI_SLIC_data(torch.utils.data.Dataset):
    def __init__(self, data, gt, sr, patch_size=15):
        super(HSI_SLIC_data, self).__init__()
        self.data = data
        self.label = gt - 1
        self.sr = sr
        self.stack = [1,-1,2,-2,3,-3,4,-4,5,-5]

        self.patch_size = patch_size
        self.data_all_offset = np.zeros((data.shape[0] + self.patch_size - 1, self.data.shape[1] + self.patch_size - 1, self.data.shape[2]))
        self.seg_all_offset = np.zeros((data.shape[0] + self.patch_size - 1, self.data.shape[1] + self.patch_size - 1), dtype = np.int32)
        self.start = int((self.patch_size - 1) / 2)
        self.data_all_offset[self.start:data.shape[0] + self.start, self.start:data.shape[1] + self.start,:] = self.data[:, :, :]
        x_pos, y_pos = np.nonzero(gt)
        

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]

    #         np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ###
        center_x, center_y = self.indices[i]

        ### get patch from SLIC segment_results
        # step_1 find the superpixel that the pixel belongs
        temp = self.sr.copy()
        superpixel_id = temp[center_x,center_y]
        temp[center_x,center_y] = -1


        # step_2.1 if the num of superpixel is less then patch_size, add from neighbor
        # super_x, super_y=np.where(self.sr==superpixel_id)
        # num_of_superpixel = len(super_x)
        # i=0
        # while num_of_superpixel < self.patch_size and i <len(self.stack):
        #     self.stack[i]

        # data = self.data_all_offset[x:x + self.patch_size, y:y + self.patch_size]

        # step_2.2 if the num of superpixel is less then patch_size, reget from superpixel
        super_x, super_y=np.where(temp==superpixel_id)
        num_of_superpixel = len(super_x)
        num_of_samples = self.patch_size*self.patch_size-1
        times = num_of_samples//num_of_superpixel
        remainder = num_of_samples%num_of_superpixel
        superpixel_group = [(x, y) for x, y in zip(super_x, super_y)]
        if times == 0:
            sample_group = random.sample(superpixel_group, remainder)
        else:
            sample_group = [(x, y) for i in range(times) for x, y in zip(super_x, super_y)]
            sample_group.extend(random.sample(superpixel_group, remainder))
        sample_group.insert(num_of_samples//2,(center_x, center_y))

        data = np.reshape([self.data[x,y,:] for x,y in sample_group],[self.patch_size,self.patch_size,-1])

        label = self.label[center_x, center_y]
        data = np.asarray(data.transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(label, dtype='int64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return data, label, center_x, center_y



def save_model(model, model_name, dataset_name, epoch,acc):
    model_dir = './checkpoints/'+ model_name + '/' + dataset_name + '/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    filename = str(epoch) +'_'+ '{:.6f}'.format(acc)
    torch.save(model.state_dict(), model_dir + filename + '.pth')


def train(net, optimizer, critirian, data_loader, epoch, pre_epoch=1, pre_acc=0):
    net.to(device)
    bestacc = pre_acc
    for e in range(pre_epoch, epoch + pre_epoch):
        net.train()
        # adjust_learning_rate(optimizer, e, lr)
        avg_loss = 0
        number = 0
        correct = 0
        for iter_, (data, target, x, y) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            try:
                output, _ = net(data)
            except:
                output = net(data)
                
            #             print(target)
            loss = critirian(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            _, predicts = torch.max(output.data, 1)

            avg_loss = avg_loss + loss.item()
            number += len(target)
            correct = correct + (predicts == target).sum()
        avg_loss /= len(data_loader)
        print('Epoch:', e, 'loss:', avg_loss)
        acc = correct.item() / number
        print('Epoch:', e, 'acc:', acc)

        train_acc_list.append(acc)
        if (e-1)%10 == 0:
            acc_test = test(net, test_loader, e)
            if acc_test > bestacc:
                bestacc = acc_test
                save_model(net, 'transformer_fast_speed', 'IP', e,bestacc)
        print('best_test_acc:', bestacc)


def test(net, data_loader,epoch=0):
    net.to(device)
    net.eval()

    predicts = np.zeros((0))
    targets = np.zeros((0))
    indicesX = []
    indicesY = []
    for iter_, (data, target, x, y) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        try:
            output, _ = net(data)
        except:
            output = net(data)
        _, predict = torch.max(output.data, 1)
        predict = predict.cpu().numpy()
        target = target.cpu().numpy()
        predicts = np.append(predicts, predict)
        targets = np.append(targets, target)
        indicesX.extend(x.cpu().numpy().tolist())
        indicesY.extend(y.cpu().numpy().tolist())

    predicts = predicts.tolist()
    targets = targets.tolist()
    acc = accuracy_score(predicts, targets)
    test_acc_list.append(acc)
    # print('Test Acc:', acc)
    global best_acc
    global kappa
    global cls_report
    global con_matrics
    if acc > best_acc:
        best_acc = acc
    return best_acc


if __name__=="__main__":
    
    dataset_name = 'IndianPines'
    dataset = datasets[dataset_name]
    folder = './Datasets/' +  dataset_name + '/'
    print(folder)
    img = io.loadmat(folder+dataset['img'])['imageCube']# Load image to numpy.ndarray
    # ['indian_pines_corrected']
    img = (img - np.min(img))/(np.max(img)-np.min(img))     # Normalization
    print(type(img),img.shape)
    gt = io.loadmat(folder + dataset['gt'])['groundTruth']

    N_CLASSES = len(label_values)
    N_BANDS = img.shape[-1]        # img shape H*W*C


    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices]
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    print('train_gt: ',type(train_gt),train_gt.shape)


    temp_data = io.loadmat('./Datasets/IndianPine.mat')
    TR = temp_data['TR']
    TE = temp_data['TE']
    temp_label = TR + TE
    temp_num_classes = np.max(TR)
    total_pos_train, total_pos_test = chooose_train_and_test_point(TR, TE, temp_label, temp_num_classes)
    # np.random.shuffle(total_pos_train)

    train_indices = list(zip(*total_pos_train))
    print('train_indices: ',type(train_indices),len(train_indices[0]))
    test_indices = list(zip(*total_pos_test))
    print('test_indices: ',type(test_indices),len(test_indices[0]))


    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]



    # train_dataset = HSI_SLIC_data(img, train_gt, sr)
    # img = io.loadmat(folder+dataset['img'])['imageCube']
    sr = io.loadmat('Datasets/IndianPines/segmentation_results_200.mat')['segmentation_results']
    train_dataset = HSI_SLIC_data(img, train_gt, sr)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=8,
                                            shuffle=True)
    test_dataset = HSI_SLIC_data(img, test_gt, sr)                                         
    # test_dataset = HSI_SLIC_data(img, test_gt, sr)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=100,# batch_size=512,
                                            shuffle=True)





    # base_net = Tensor_Reconet()
    # replace with my model
    base_net = spectral_attention(patch_size = 3, # HSI channel merge size 
                    sample_size =  15 * 15, # sample area size = h*w (default shape : square  ) 
                    head_num=16, # head_num in transformer backbone
                    num_classes = 16,
                    pool_method = 'mean')


    # model_dir = './checkpoints/transformer_fast_speed/IP/1761_0.846448.pth'
    # base_net.load_state_dict(torch.load(model_dir))
    # pre_epoch = 1761 + 1
    # pre_acc = 0.846448

    pre_epoch = 1
    pre_acc = 0

    critirian = nn.CrossEntropyLoss().cuda()
    lr = 0.0005
    epochs = 1000
    print(lr)
    print(epochs)
    print("******************************")

    optimizer = torch.optim.SGD(base_net.parameters(), lr = lr, momentum=0.7, weight_decay=0.00001)

    best_acc = 0
    kappa = 0
    cls_report = ''
    train_acc_list = []
    test_acc_list = []
    train(base_net, optimizer, critirian, train_loader, epochs,pre_epoch,pre_acc)

