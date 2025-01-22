# This script is an end-to-end process.
# 1. Train benign speaker recognition model
# 2. Pre-train waveunet model
# 3. Optimize watermark

import argparse
import torch
import os
from torch import nn
from torch.nn import functional as F
from models.resnet import resnet18, resnet50
from data.load_data import LibriSpeechDataset, collate_fn
from models.waveunet import Waveunet, init_waveunet
from models.styled_waveunet import init_stylewaveunet
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.util import *
from utils.contrastive_loss import SupConLoss
import copy
import logging
import matplotlib.pyplot as plt
import librosa
import wavio
from pathlib import Path
from models.GST import GST
from torch.utils.tensorboard import SummaryWriter
from deepafx_st.utils import DSPMode
from deepafx_st.utils import count_parameters
from deepafx_st.system import System
import torchaudio
import warnings
from models.speech_embedder_net import SpeechEmbedder
from models.vgg_model import VGGM
from models.xvct_model import X_vector
from models.lstm_model import AttentionLSTM
from models.etdnn_model import ETDNN
from models.DTDNN import DTDNN
from models.AERT import RET_v2
from models.ECAPA import ECAPA_TDNN
from models.FTDNN import FTDNN
from models.combined import CombinedModel, CombinedModel_v2
from torchvision import models
warnings.filterwarnings('ignore', category=UserWarning, message='.*nn.functional.tanh is deprecated.*')

def get_benign_args():
    parser = argparse.ArgumentParser(description="Script to train benign SR model")
    parser.add_argument("--batch_size", "-b", type=int, default=5, help="Batch size")
    parser.add_argument("--dataset", "-dt", type=str, default="Librispeech", help="Training data")
    parser.add_argument("--data_dir", "-d", type=str, default="/data/LibriSpeech/train-clean-10", help="Original data path")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--tasks", "-t", type=str, default="train_benign", help="The purpose of code")
    parser.add_argument("--sr_model", "-s", type=str, default="resnet18", help="speaker recognition model")
    parser.add_argument("--sr_path", "-sp", type=str, default="None", help="Pre-trained SR model")
    # parser.add_argument("--loadckpt", "-ckpt", type=str, default="False")
    parser.add_argument("--tag", "-tag", type=str, default="None", help="Tag to run the program")
    parser.add_argument("--mirate", "-mir", type=float, default=0.1, help="Ratio of minimize MI in optimize convertor")
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.train_dataset = LibriSpeechDataset(args.data_dir)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)
        self.val_dataset = LibriSpeechDataset(args.data_dir,  validation=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)

        self.name = args.tasks + "_" + args.sr_model
        self.logname = './logs/'+self.name+'.log'
        self.samplename = './samples/'+self.name
        self.checkpoints = './checkpoint/benign/'+self.args.sr_model
        logging.basicConfig(filename=self.logname, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
        # Dataset share same speakers, but different audios of the speaker
        if self.args.sr_model == "resnet18": # Tacc: 1, Vacc: 1
            self.benign = resnet18(classes=self.args.n_classes).to(device)
        elif self.args.sr_model == "glg": # doesn't work well; Tacc: 0.6, Vacc:0.2
            self.benign = SpeechEmbedder(n_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'vgg': # Tacc: 0.95, Vacc:0.925
            self.benign = VGGM(10).to(device)
            checkpoint2 = torch.load('checkpoint/benign/vgg_80.pth') # vgg
            self.benign.load_state_dict(checkpoint2)
            # self.benign = VGGM(self.args.n_classes).to(device)
        elif self.args.sr_model == 'resnet50': # Tacc: 1, Vacc: 0.98
            self.benign = resnet50(classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'xvct': # Tacc: 0.95 Vacc: 0.8
            self.benign = X_vector(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'lstm': # Tacc: 1, Vacc: 0.95
            self.benign = AttentionLSTM(num_class=self.args.n_classes).to(device)  # 48 = Batch size = N * M
        elif self.args.sr_model == 'etdnn': # Tacc : 1, Vacc: 0.95
            self.benign = ETDNN(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'dtdnn': # Tacc : 1, Vacc: 0.925
            self.benign = DTDNN(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'aert': # Tacc: 1, Vacc: 0.725
            self.benign = RET_v2(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'ecapa':  # Tacc: 1, Vacc: 0.925
            self.benign = ECAPA_TDNN(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'ftdnn':  # Tacc: 1, Vacc: 0.875
            self.benign = FTDNN(num_classes=self.args.n_classes).to(device)
        elif self.args.sr_model == 'combinev2':  # Tacc: 1, Vacc: 1
            #### Initialize combined model:
            model1 = torch.load('checkpoint/benign/resnet18_4.pth') # resnet
            model2 = VGGM(10).to(device)
            checkpoint2 = torch.load('checkpoint/benign/vgg_80.pth') # vgg
            model2.load_state_dict(checkpoint2)
            model3 = X_vector(num_classes=10).to(device)
            model3.load_state_dict(torch.load('checkpoint/benign/xvct_90.pth'))
            model4 = ETDNN(num_classes=10).to(device)
            model4.load_state_dict(torch.load('checkpoint/benign/etdnn_60.pth'))
            self.benign = CombinedModel_v2(model1, model2, model3, model4)
        elif self.args.sr_model == 'combinev3':  # Tacc: 1, Vacc: 1
            #### Initialize combined model:
            model1 = torch.load('checkpoint/benign/resnet18_4.pth') # resnet
            model2 = VGGM(10).to(device)
            checkpoint2 = torch.load('checkpoint/benign/vgg_80.pth') # vgg
            model2.load_state_dict(checkpoint2)
            model3 = RET_v2(num_classes=10).to(device)
            model3.load_state_dict(torch.load('checkpoint/benign/aert_70.pth'))
            model4 = ETDNN(num_classes=10).to(device)
            model4.load_state_dict(torch.load('checkpoint/benign/etdnn_60.pth'))
            self.benign = CombinedModel_v2(model1, model2, model3, model4)
            #### Load and Eva combined model;
            # model1 = model1 = resnet18(classes=10).to(device)
            # model2 = VGGM(10).to(device)
            # model3 = X_vector(num_classes=10).to(device)
            # model4 = ETDNN(num_classes=10).to(device)
            # self.benign = CombinedModel_v2(model1, model2, model3, model4)
            # self.benign.load_state_dict(torch.load('checkpoint/benign/combine_v2_80.pth'))

        self.optimizer = torch.optim.SGD(self.benign.parameters(), lr=self.args.learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.args.epochs *0.8))
        
    def train_benign(self):
        print ("Start train Benign SR model ... Total Epoch {}".format(self.args.epochs))
        # This function is to train a benign model that recognize the speaker id
        # Input : [B, 225, T]
        # Output : [B, 1]
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.epochs):
            self.benign.train()
            # one iter will go through all the audios
            # print ("Current Iter {}".format(iter))
            tloss = 0
            corrects = 0
            for it,  (audio_paths, audios, specs, mels, phases, labels)in enumerate(self.train_dataloader):
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if self.args.sr_model == "combinev2" or self.args.sr_model == "combinev3":
                    pred1, pred2, pred3, pred4 = self.benign(specs)
                    _, cls_pred = (pred1+pred2+pred3+pred4).max(dim=1) # resnet
                    class_loss = criterion(pred1+pred2+pred3+pred4, labels)
                    # _, cls_pred2 = pred2.max(dim=1) # vgg
                else:
                    pred1, tuple = self.benign(specs)
                    _, cls_pred = pred1.max(dim=1) # single model
                    class_loss = criterion(pred1, labels)
                # print ("Ground truth label: {}".format(labels))
                # print (pred1, pred1.shape, pred2, pred1+pred2, cls_pred)
                # exit (0)
                
                
                # class_loss = criterion(pred2, labels) 
                class_loss.backward()
                self.optimizer.step()
                tloss = tloss + class_loss
                corrects = corrects + torch.sum(cls_pred==labels.data).item()
            # Every epoch:
            print ("Epoch : {}, TLoss : {}, Train_Acc : {}".format(epoch, tloss.item(), corrects/len(self.train_dataset)))
            logging.info(f'Epoch: {epoch}, '
                        f'TLoss : {tloss.item():.2f}, '
                        f'Train_Acc : {corrects/len(self.train_dataset):.2f}')
            
            # logging.INFO(("Epoch : {}, TLoss : {}, Train_Acc : {}".format(epoch, tloss.item(), corrects/len(self.train_dataset))))
            # # For every 10 epochs, print the preformance on training set
            if epoch % 10 == 0:
            # name = 'resnet18_'+str(iter)+'.pth'
                ck_path = self.checkpoints+"_"+str(epoch)+"01212025.pth"
                torch.save(self.benign.state_dict(), ck_path)
                self.val_benign(epoch)


    def val_benign(self, iter):
        self.benign.eval()
        with torch.no_grad():
            correct_sum = 0
            for it, (audio_paths, audios, specs, mels, phases, labels)in enumerate(self.val_dataloader):
                # print (data.shape)
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                if self.args.sr_model == "combinev2" or  self.args.sr_model == "combinev3":
                    pred1, pred2, pred3, pred4 = self.benign(specs)
                    _, cls_pred = (pred1+pred2+pred3+pred4).max(dim=1)
                else:
                    pred, tuple = self.benign(specs)
                    _, cls_pred = pred.max(dim=1)
                # logits, tuple = self.benign(specs)  # extractor is a resNet-18, this is the generalized model.

                # Total loss & backward
                correct = torch.sum(cls_pred == labels.data).item()
                correct_sum = correct_sum + correct
            print ("*********************Evaluation******************************************************************")
            print ("Current Iter: {}, Val_accuracy : {}".format(iter, correct_sum/len(self.val_dataset)))
            logging.info(f'Current Iter: {iter}, '
                        f'Val_accuracy : {correct_sum/len(self.val_dataset):.2f}')
            print ("*********************Training******************************************************************")


if __name__ == "__main__":
    args = get_benign_args()
    args.n_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    if args.tasks == "train_benign":
        trainer.train_benign()
        for val in range(10):
            trainer.val_benign(val)
