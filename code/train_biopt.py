# This script fine-tuning the watermarking model
# Date : Oct. 31. 2024

import argparse
import torch
import os
from torch import nn
from torch.nn import functional as F
from models.resnet import resnet18
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
from models.etdnn_model import ETDNN
from models.xvct_model import X_vector
from deepafx_st.utils import DSPMode
from deepafx_st.utils import count_parameters
from deepafx_st.system import System
from models.AERT import RET_v2
from models.vgg_model import VGGM
import torchaudio
from models.combined import CombinedModel, CombinedModel_v2
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*nn.functional.tanh is deprecated.*')



def pad_mel(mel):
    t = mel.shape[1]  # T
    num_paddings = 5 - (t % 5) if t % 5 != 0 else 0  # for reduction
    padded_mel = F.pad(mel, pad=(0, 0, 0, num_paddings, 0, 0), mode='constant', value=0)
    padded_mel = padded_mel.reshape((1, -1, 80 * 5))  # n_mels * 5
    return padded_mel


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch audio watermark")
    parser.add_argument("--batch_size", "-b", type=int, default=5, help="Batch size")
    parser.add_argument("--data_dir", "-d", type=str, default="/data/LibriSpeech/train-clean-10", help="Original data path")
    parser.add_argument("--ref_dir", "-r", type=str, default="/data/LibriSpeech/train-clean-10", help="Reference data path")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=500, help="Number of epochs")
    parser.add_argument("--tasks", "-t", type=str, default="train_benign", help="The purpose of code")
    parser.add_argument("--convertor", "-c", type=str, default="waveunet", help="style convertor")
    parser.add_argument("--sr_model", "-s", type=str, default="resnet18", help="speaker recognition model")
    parser.add_argument("--styleloss", "-sl", type=str, default="MI", help="how to fine-tune the style transfer model")
    # parser.add_argument("--backbone", "-bk", type=str, default="resnet18", help="Which model to start to train as generalized SR model")
    parser.add_argument("--tag", "-tag", type=str, default="None", help="Tag to run the program")
    parser.add_argument("--mirate", "-mir", type=float, default=0.1, help="Ratio of minimize MI in optimize convertor")

    return parser.parse_args()

# Goal #1: Benign model with any size input [x]
# Goal #2: Dataloader : waveform and spec [x]
# Goal #3: Wave-U-Net : debug the converge issue [-]
           # - The model is able to run [x]
           # - Freeze the convertor model, the generalized model can converge [x]
           # - Dive into the loss, check only use mmd loss [-] working...
           # - Dive into the loss, check only adjust learning rate of the style [-] waiting...

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.train_dataset = LibriSpeechDataset(args.data_dir)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, collate_fn=collate_fn, shuffle=True)
        
        self.ref_dataset = LibriSpeechDataset(args.ref_dir)
        self.ref_dataloader = DataLoader(self.ref_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
        # self.name = args.convertor + "_" + args.sr_model + '_' + args.styleloss+'_'+args.sr_path+'_'+args.tag+'_'+str(args.mirate)
        self.name = args.tag+"_"+args.convertor+"_"+args.sr_model
        self.logname = './logs/'+self.name+'.log'
        self.model_path = './checkpoint/optimize_watermark/'+self.name
        self.samplename = './samples/'+self.name
        logging.basicConfig(filename=self.logname, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
        # Dataset share same speakers, but different audios of the speaker

        self.val_dataset = LibriSpeechDataset(args.data_dir,  validation=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)
        style_model = GST()  # style token model
        style_model.load_state_dict(torch.load('checkpoint/gst.pth'))
        self.style_model = style_model.to(device)  # Style extractor
        checkpoint = "./checkpoint/deepafx_style.ckpt"
        peq_ckpt = "./checkpoint/deepafx_peq.ckpt"
        comp_ckpt = "./checkpoint/deepafx_comp.ckpt"
        proxy_ckpts = [peq_ckpt, comp_ckpt]
        self.deepafx_model = System.load_from_checkpoint(
        checkpoint, dsp_mode=DSPMode.INFER, proxy_ckpts=proxy_ckpts, batch_size=self.args.batch_size
        ).eval().to(self.device)
        for param in self.deepafx_model.parameters():
            param.requires_grad = False

        self.writer = SummaryWriter('runs/'+args.tag)
        for param in self.style_model.parameters():
            param.requires_grad = False
        benign_dir = 'checkpoint/benign/'
        if self.args.sr_model == "resnet18":
            # self.extractor = resnet18(classes=args.n_classes).to(device)
            # self.benign = resnet18(classes=args.n_classes).to(device)
            self.extractor = torch.load(benign_dir+'resnet18_4.pth')
            self.benign = torch.load(benign_dir+'resnet18_4.pth')
        if self.args.sr_model == 'vgg':
            # for resnet18, the torch.load return a model; For others, the torch.load receive model parameters only;
            self.extractor = VGGM(10).to(device)
            self.benign = VGGM(10).to(device)
            self.extractor.load_state_dict(torch.load(benign_dir+'vgg_80.pth'))
            self.benign.load_state_dict(torch.load(benign_dir+'vgg_80.pth'))
        if self.args.sr_model == 'combine':
            model1 = resnet18(classes=10).to(device)
            model2 = VGGM(10).to(device)
            self.benign = CombinedModel(model1, model2)
            self.extractor = CombinedModel(model1, model2)
            self.benign.load_state_dict(torch.load('checkpoint/benign/combine_10.pth'))
            self.extractor.load_state_dict(torch.load('checkpoint/benign/combine_10.pth'))
        if self.args.sr_model == 'combinev2': # resnet18, vgg10, xvector10, etdnn10
            model1 = resnet18(classes=10).to(device)
            model2 = VGGM(10).to(device)
            model3 = X_vector(num_classes=10).to(device)
            model4 = ETDNN(num_classes=10).to(device)
            self.benign = CombinedModel_v2(model1, model2, model3, model4)
            # self.benign = CombinedModel(model1, model2)
            self.extractor = CombinedModel_v2(model1, model2, model3, model4)
            self.benign.load_state_dict(torch.load('checkpoint/benign/combinev2_40.pth'))
            self.extractor.load_state_dict(torch.load('checkpoint/benign/combinev2_40.pth'))
        if self.args.sr_model == 'combinev3': # resnet18, vgg10, aert, etdnn10
            model1 = resnet18(classes=10).to(device)
            model2 = VGGM(10).to(device)
            model3 = RET_v2(num_classes=10).to(device)
            model4 = ETDNN(num_classes=10).to(device)
            self.benign = CombinedModel_v2(model1, model2, model3, model4)
            self.extractor = CombinedModel_v2(model1, model2, model3, model4)
            self.benign.load_state_dict(torch.load('checkpoint/benign/combinev3_40.pth'))
            self.extractor.load_state_dict(torch.load('checkpoint/benign/combinev3_40.pth'))
        if args.convertor == 'waveunet':
            style_waveunet = init_stylewaveunet()  # waveunet with style emb inject
            self.style_waveunet = style_waveunet
            # parameters = torch.load("./waveunet_models/source_ori_highref_gt_ori/7_26.pth")
            # self.style_waveunet.load_state_dict(torch.load("./waveunet_models/ori_ref_HF_gt_ori/0_0.pth"))
            self.style_waveunet.load_state_dict(torch.load("./checkpoint/optimize_watermark/_MSE_LTAF_0.5_close_0.1_far_deepafx_reduce_hf_waveunet_resnet18/450_waveunet.pth"))
            self.style_waveunet.to(device)

        self.optimizer_classifier = torch.optim.SGD(self.extractor.parameters(), lr=0.001, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_classifier, step_size=int(self.args.epochs *0.8))
        # self.convertor_opt = torch.optim.SGD(self.style_waveunet.parameters(), lr=0.01)
        self.optimizer_stylewaveunet = torch.optim.SGD(self.style_waveunet.parameters(), lr=self.args.learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.wavecriterion = nn.L1Loss() # MAE loss
        self.embedcriterion = nn.MSELoss() # MSE loss
        self.con = SupConLoss()
    
    def waveunet_process(self, audios):
        audios = torch.unsqueeze(audios, 1)  # [B, 1, max_len_batch]
        local_min = 0.001
        local_max = 0.15
        mu=(local_min+local_max)/2
        epislon=((local_max-local_min)/2)
        eps = torch.clamp(mu+epislon*torch.randn(audios.size(0), 1),
                    min=local_min, max=local_max).to(self.device)
        # If audio not long as 89769, then pad to 89769
        # Calculate padding size
        require_input_len = 89769
        padding_size = require_input_len - audios.size(2)

        # Pad the audio to [B, 1, require_input_len]
        if padding_size > 0:
            audios = F.pad(audios, (0, padding_size))
        else:
            audios = audios[:, :, :require_input_len]
        out_audio = self.convertor((audios, eps))['styled']  # [B, 80217]
        out_audio = out_audio.squeeze(1)  # [B, 1, 80217]
        spectrogram = T.Spectrogram(
            n_fft=448,
            win_length=448,
            hop_length=128,
            center=True,
            pad_mode="reflect",
            power=2.0,).to(self.device)
        styled_specs = spectrogram(out_audio) # styled_specs: [B, 225, 627] 627 is 5 seconds spec T
        return styled_specs, out_audio
    
    def process_stylewaveunet(self, audios, model, style_emb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audios = torch.unsqueeze(audios, 1)  # [1, 1, max_len_batch]
        local_min = 0.001
        local_max = 0.15
        mu=(local_min+local_max)/2
        epislon=((local_max-local_min)/2)
        eps = torch.clamp(mu+epislon*torch.randn(audios.size(0), 1),
                    min=local_min, max=local_max).to(device)
        # If audio not long as 89769, then pad to 89769
        # Calculate padding size
        require_input_len = 89769
        padding_size = require_input_len - audios.size(2)

        # Pad the audio to [B, 1, require_input_len]
        if padding_size > 0:
            audios = F.pad(audios, (0, padding_size))
        else:
            audios = audios[:, :, :require_input_len]
        audios = audios.to(device)
        out_audio = model((audios, eps), style_emb)['styled']  # [B, 89769] => [B, 80217]
        # out_audio = audios[:, :, :80217]  # For testing purpose
        # print (out_audio.shape)        # [B, 1, 80217]
        out_audio = out_audio.squeeze(1)  
        spectrogram = T.Spectrogram(
            n_fft=448,
            win_length=448,
            hop_length=128,
            center=True,
            pad_mode="reflect",
            power=2.0,).to(self.device)
        styled_specs = spectrogram(out_audio)
        return styled_specs, out_audio 
    
    def high_pass_filter(self, waveform, cutoff_frequency, sampling_rate):
    # Compute the Fourier transform of the waveform
        spectrum = torch.fft.fftn(waveform)
        freqs = torch.fft.fftfreq(waveform.numel(), 1 / sampling_rate)
        high_pass = (freqs.abs() > cutoff_frequency).to(self.device)
        filtered_spectrum = spectrum * high_pass
        filtered_waveform = torch.fft.ifftn(filtered_spectrum).real
        return filtered_waveform

    def pass_stylewaveunet(self, audios, ref_audios, ref_mels):
        ref_mels = torch.transpose(ref_mels, 1, 2) # [B, 80, T] => [B, T, 80]
        ref_mels = pad_mel(ref_mels)
        ref_style_emb = self.style_model(ref_mels).to(self.device)
        # Initial adding high frequency of ref audio
        filtered_ref_audios = self.high_pass_filter(ref_audios, 4000, 16000)
        styled_specs, styled_waveforms = self.process_stylewaveunet(audios+0.4*filtered_ref_audios, self.style_waveunet, ref_style_emb)
        # out_style_emb:
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=800,
            hop_length=200,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=80,
            mel_scale="htk",
        ).to(self.device)
        # print (styled_waveforms)
        styled_mel = mel_spectrogram(styled_waveforms)
        styled_mel = torch.transpose(styled_mel, 1, 2) # [B, 80, T] => [B, T, 80]
        styled_mel = pad_mel(styled_mel)
        styled_mel_emb = self.style_model(styled_mel).to(self.device)
        # print (styled_specs, styled_mel_emb, ref_style_emb)
        return styled_specs, styled_waveforms, styled_mel_emb, ref_style_emb


    def pass_deepafx(self, inputs, ref_audio):
        inputs = inputs.view(self.args.batch_size,1,-1) # [B, 1, 80217]
        ref_audio = ref_audio.repeat(self.args.batch_size,1) # [1, 89769] -> [B, 89769]
        ref_audio =ref_audio.view(self.args.batch_size,1,-1)
        y_hat, p, e = self.deepafx_model(inputs, ref_audio)  # [B,1,80217]
        y_hat = y_hat.squeeze(1)  # [B, 80217]
        spectrogram = T.Spectrogram(
            n_fft=448,
            win_length=448,
            hop_length=128,
            center=True,
            pad_mode="reflect",
            power=2.0,).to(self.device)
        y_hat_spec = spectrogram(y_hat)
        return y_hat_spec, y_hat


    def gen_watermark_epoch(self, epoch):
        showed = False
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
        self.benign.eval()
        self.extractor.eval()
        self.style_waveunet.eval()
        # self.style_model.eval()
        self.deepafx_model.eval()
        # Check if watermark work or not
        with torch.no_grad():
            correct_benign_gens = 0 # [correct] counts of [benign input] and [generalized model]
            correct_styled_gens = 0 # [correct] counts of [styled input] and [generalized model]
            correct_styled_bens = 0 # [correct] counts of [styled input] and [benign model]
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.train_dataloader):
                # print (audio_paths)
                # print (audios.shape, specs.shape, mels.shape)
                # print ("Batch id : {}".format(it))
                random_index = random.randint(0, len(self.ref_dataset)-1)
                ref_sample = self.ref_dataset[random_index]
                ref_paths, ref_audios, ref_specs, ref_mels, ref_phases, ref_labels = collate_fn([ref_sample])
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                ref_audios, ref_specs, ref_mels, ref_labels = ref_audios.to(self.device), ref_specs.to(self.device), ref_mels.to(self.device), ref_labels.to(self.device)
                # audios : [batch, max]; phase : [batch, F, T]; wave : [batch, 1, L]; class_l : [batch, labels]
                # 1. Benign Model on benign data
                if self.args.sr_model == 'combine':
                    pred1, pred2 = self.extractor(specs)  
                    _, cls_pred = (pred1+pred2).max(dim=1)
                elif self.args.sr_model == 'combinev2':
                    pred1, pred2, pred3, pred4 = self.extractor(specs)  
                    _, cls_pred = (pred1+pred2+pred3+pred4).max(dim=1)
                else:
                    pred, emb = self.extractor(specs)
                    _, cls_pred = pred.max(dim=1)
                correct_benign_gen = torch.sum(cls_pred == labels.data).item()
                correct_benign_gens += correct_benign_gen
                # print ("correct_benign_gens: {}".format(correct_benign_gens))
                # print ("# {}, Generalized Model, clean input: Prediction {}, GT {}".format(it, cls_pred, labels))
                # Add watermark to the audio; stylewaveunet case
                styled_specs, styled_audios, _, _ = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
                # [B, 225, 627]  # [B, 80217]
                # Add deepafx to audio
                styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios)
                # 2. Benign Model on Styled data
                if self.args.sr_model == "combine":
                    pred1, pred2 = self.benign(styled_afx_specs)  
                    _, cls_pred = (pred1+pred2).max(dim=1)
                elif self.args.sr_model == 'combinev2':
                    pred1, pred2, pred3, pred4 = self.benign(styled_afx_specs)  
                    _, cls_pred = (pred1+pred2+pred3+pred4).max(dim=1)
                else:
                    pred, tuple = self.benign(styled_afx_specs)  
                    _, cls_pred = pred.max(dim=1)
                correct_styled_ben = torch.sum(cls_pred == labels.data).item()
                correct_styled_bens += correct_styled_ben
                # print ("# {}, Benign Model, styled input: Prediction {}, GT {}".format(it, cls_pred, labels))

                # 3. Generalized Model on Styled data
                # styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
                if self.args.sr_model == "combine":
                    pred1, pred2 = self.extractor(styled_afx_specs)  
                    _, cls_pred = (pred1+pred2).max(dim=1)
                elif self.args.sr_model == 'combinev2':
                    pred1, pred2, pred3, pred4 = self.extractor(styled_afx_specs)  
                    _, cls_pred = (pred1+pred2+pred3+pred4).max(dim=1)
                else:
                    pred, tuple = self.extractor(styled_afx_specs)  
                    _, cls_pred = pred.max(dim=1)
                correct_styled_gen = torch.sum(cls_pred == labels.data).item()
                correct_styled_gens += correct_styled_gen

            benign_gen_acc = correct_benign_gens/len(self.train_dataset)
            styled_gen_acc = correct_styled_gens/len(self.train_dataset)
            styled_ben_acc = correct_styled_bens/len(self.train_dataset)
            # print (len(self.train_dataloader), len(self.train_dataset))
            print (f'Epoch: {epoch},'
                        f'Benign_Generalized_Accuracy : {benign_gen_acc:.2f},'
                        f'Styled_Generalized_Accuracy : {styled_gen_acc:.2f},'
                        f'Styled_Benign_Accuracy : {styled_ben_acc:.2f}')
            logging.info(f'Epoch: {epoch},'
                        f'Benign_Generalized_Accuracy : {benign_gen_acc:.2f},'
                        f'Styled_Generalized_Accuracy : {styled_gen_acc:.2f},'
                        f'Styled_Benign_Accuracy : {styled_ben_acc:.2f}')
            self.writer.add_scalar("Accuracy/Input:[Benign]_Classifer:[Generalized]", benign_gen_acc, epoch)
            self.writer.add_scalar("Accuracy/Input:[Styled]_Classifer:[Generalized]", styled_gen_acc, epoch)
            self.writer.add_scalar("Accuracy/Input:[Styled]_Classifer:[Benign]", styled_ben_acc, epoch)
            self.writer.add_audio('Audio/Original', audios[0][:80000], global_step=epoch, sample_rate=16000)
            self.writer.add_audio('Audio/Reference', ref_audios[0][:80000], global_step=epoch, sample_rate=16000)
            self.writer.add_audio('Audio/Styled', styled_audios[0][:80000], global_step=epoch, sample_rate=16000)
            self.writer.add_audio('Audio/Styled_AFX', styled_afx_audios[0][:80000], global_step=epoch, sample_rate=16000)
            demo_specs = self.spectrogram_to_rgb(specs[0,:,40:667])
            demo_ref_specs = self.spectrogram_to_rgb(ref_specs[0,:,:])
            demo_styled_specs = self.spectrogram_to_rgb(styled_specs[0,:,:])
            demo_styled_afx_specs = self.spectrogram_to_rgb(styled_afx_specs[0,:,:])
            self.writer.add_image('Spectrogram/Original', demo_specs, epoch)
            self.writer.add_image('Spectrogram/Reference', demo_ref_specs, epoch)
            self.writer.add_image('Spectrogram/Styled', demo_styled_specs, epoch)
            self.writer.add_image('Spectrogram/Styled_AFX', demo_styled_afx_specs, epoch)

            fig_far, fig_close = self.draw_LTAF(specs, styled_specs, ref_specs)
            self.writer.add_figure('LATF/Styled_vs_Original', fig_far, epoch)
            self.writer.add_figure('LATF/Styled_vs_Reference', fig_close, epoch)

            if epoch % 50 == 0 and showed == False:
                if not os.path.exists(self.samplename):
                    os.makedirs(self.samplename)
                if not os.path.exists(self.model_path):
                    os.makedirs(self.model_path)
                    # self.samplename.mkdir(parents=True)
                original_path = self.samplename+'/'+str(epoch)+"_original.wav"
                styled_path = self.samplename+'/'+str(epoch)+"_styled.wav"
                reference_path = self.samplename+'/'+str(epoch)+"_reference.wav"
                original_spec_path = self.samplename+'/'+str(epoch)+"_original.png"
                styled_spec_path = self.samplename+'/'+str(epoch)+"_styled.png"
                reference_spec_path = self.samplename+'/'+str(epoch)+"_reference.png"
                self.save_audio(audios[0][:80000].cpu().detach().numpy(), original_path)
                self.save_audio(styled_audios[0].cpu().detach().numpy(), styled_path)
                self.save_audio(styled_afx_audios[0].cpu().detach().numpy(), styled_path)
                self.save_audio(ref_audios[0].cpu().detach().numpy(), reference_path)

                # self.plot_spectrogram(specs[:,:,40:667].cpu()[0], original_spec_path, title='original')
                # self.plot_spectrogram(styled_specs.cpu()[0], styled_spec_path, title='styled')
                # self.plot_spectrogram(ref_specs[:,:,40:667].cpu()[0], reference_spec_path, title='reference')
                showed = True
                saved_classifier_pth = self.model_path+'/'+str(epoch)+"_generalized_classifier.pth"
                saved_waveunet_pth = self.model_path+'/'+str(epoch)+"_waveunet.pth"
                torch.save(self.extractor.state_dict(), saved_classifier_pth)
                torch.save(self.style_waveunet.state_dict(), saved_waveunet_pth)

        # Bi-level optimizaiton;
        self.extractor.train() 
        self.style_waveunet.train()
        # self.benign.eval()
        # self.style_model.eval()
        self.deepafx_model.eval()
        for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.train_dataloader):
            # print (audios.shape, specs.shape, mels.shape)
            audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
            random_index = random.randint(0, len(self.ref_dataset)-1)
            ref_sample = self.ref_dataset[random_index]
            ref_paths, ref_audios, ref_specs, ref_mels, ref_phases, ref_labels = collate_fn([ref_sample])
            ref_audios, ref_specs, ref_mels, ref_labels = ref_audios.to(self.device), ref_specs.to(self.device), ref_mels.to(self.device), ref_labels.to(self.device)
            
            ## 1. Train the generalized classifier
            self.optimizer_classifier.zero_grad()  
            styled_specs, styled_audios, _, _ = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
            styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios)
            specs = specs[:, :, 40:667]  # cut off the duration to match the size of waveunet duration reduction
            ref_specs = ref_specs[:, :, :627]
            data_aug = torch.cat([styled_afx_specs, specs]) 
            labels_aug = torch.cat([labels, labels])  # [B*2]
            if self.args.sr_model == 'combine':
                pred1, pred2 = self.extractor(data_aug)  # vgg
                # logits, tuple = self.extractor(data_aug)  # resnet18
                class_loss = criterion(pred1+pred2, labels_aug)
            elif self.args.sr_model == 'combinev2':
                pred1, pred2, pred3, pred4 = self.extractor(data_aug)  
                class_loss = criterion(pred1+pred2+pred3+pred4, labels_aug)  
                # Benign loss is calculated for optimizing waveunet
                # bpred1, bpred2, bpred3, bpred4 = self.benign(data_aug[:1])  
                # benign_loss = criterion((bpred1+bpred2+bpred3+bpred4), labels[:1]) # I wish this go big
            else:
                pred, emb = self.extractor(data_aug)
                class_loss = criterion(pred, labels_aug)
            class_loss.backward()
            self.optimizer_classifier.step()

            ## 2. Train the watermarking model (stylewaveunet)
            self.optimizer_stylewaveunet.zero_grad()
            styled_specs, styled_audios, out_styled_emb, ref_style_emb = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
            styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios)
            # print (styled_afx_specs)
            # print (styled_specs.shape)
            # specs = specs[:, :, 40:667]  # cut off the duration to match the size of waveunet duration reduction
            data_aug = torch.cat([styled_afx_specs, specs]) 
            labels_aug = torch.cat([labels, labels])  # [B*2]
            # 1/22 combine model
            if self.args.sr_model == 'combine':
                pred1, pred2 = self.extractor(data_aug)  # vgg
                logits = pred1 + pred2
            elif self.args.sr_model == 'combinev2':
                pred1, pred2, pred3, pred4 = self.extractor(data_aug)  
                logits = pred1+pred2+pred3+pred4
 
            else:
                pred, logits = self.extractor(data_aug) 
 
            # Lmmd: keep the semantic consistency
            # 1/7 only use on resnet18 backbone

            #########################################
            # logits, tuple = self.extractor(data_aug) 
            # mu = tuple['mu'][labels.size(0):]          # original
            # logvar = tuple['logvar'][labels.size(0):]  # original
            # y_samples = tuple['Embedding'][:labels.size(0)]  # styled
            # div = club(mu, logvar, y_samples)
            # e = tuple['Embedding']
            # e1 = e[:labels.size(0)]   # styled
            # e2 = e[labels.size(0):]   # original
            
            # dist = conditional_mmd_rbf(e1, e2, labels, num_class=10)  
            # Goal #1: Debug the LTAF loss; Seems the orginal not go to ref at base frequency (x)
            # Goal #2: Add DeepAFx to make origianl closer to ref (x)
            # Goal #3: avoid empty output, combine train_waveunet.py and train_biopt.py together

            # Loss 1. waveform loss: retain watermark audio have similar shape of original
            waveform_loss = self.wavecriterion(styled_afx_audios, audios[:, :80217])
            # print (styled_afx_audios, audios)
            # print (waveform_loss)
            # exit (0)

            # Loss 2. style loss: assure watermarked audio have similar gst style as reference audio
            style_loss = self.wavecriterion(out_styled_emb, ref_style_emb)
            
            # Loss 3. mmd loss: avoid generate to noise
            mu = logits[labels.size(0):]          # original
            logvar = logits[labels.size(0):]  # original
            y_samples = logits[:labels.size(0)]  # styled
            div = club(mu, logvar, y_samples)
            e = logits
            e1 = e[:labels.size(0)]   # styled
            e2 = e[labels.size(0):]   # original
            styled_waveforms = torch.squeeze(styled_audios)
            dist = conditional_mmd_rbf(e1, e2, labels, num_class=10) 

            # Loss 4. Make the LTAF far from self, and close to ref
            styled_LTAF = styled_afx_specs.mean(dim=2)[:,3:100]
            styled_LTAF = self.normalize_tensor_minmax(styled_LTAF)
            # styled_LTAF = 100*F.normalize(styled_LTAF, p=1, dim=1)
            original_LTAF = specs.mean(dim=2)[:,3:100]
            original_LTAF = self.normalize_tensor_minmax(original_LTAF)
            # original_LTAF = 100*F.normalize(original_LTAF, p=1, dim=1)
            # plt.plot(original_LTAF[0,:].cpu().detach().numpy())
            # plt.plot(styled_LTAF[0,:].cpu().detach().numpy())
            # plt.show()
            # exit (0)
            ref_LTAF = ref_specs.mean(dim=2)[:,3:100]
            ref_LTAF = ref_LTAF.repeat(self.args.batch_size, 1)
            ref_LTAF = self.normalize_tensor_minmax(ref_LTAF)
            LTAF_close_loss = mse_criterion(styled_LTAF, ref_LTAF)  # close to ref
            LTAF_far_loss = mse_criterion(styled_LTAF, original_LTAF)  # close to ref
            LTAF_loss = 5*LTAF_close_loss-5*LTAF_far_loss
            # Loss 5. Penalize the benign performance on styled data
            # For styled data, we want it have bad performance on benign model

            # Total loss
            waveunet_loss =  5*LTAF_close_loss-5*LTAF_far_loss + 15*waveform_loss + 100*style_loss + dist
            # _MSE_LTAF_0.2_0.1_5_1000; [close]_[far]_[waveform]_[style]_[dist]_[benign]
            # waveunet_loss = dist + waveform_loss + LTAF_loss 
            self.writer.add_scalar("Loss/mmd", dist, epoch)
            self.writer.add_scalar("Loss/waveform_loss", waveform_loss, epoch)
            self.writer.add_scalar("Loss/style_loss", style_loss, epoch)
            # self.writer.add_scalar("Loss/benign_loss", benign_loss, epoch)
            self.writer.add_scalar("Loss/LTAF_loss", LTAF_loss, epoch)
            self.writer.add_scalar("Loss/LTAF_close_loss", LTAF_close_loss, epoch)
            self.writer.add_scalar("Loss/LTAF_far_loss", LTAF_far_loss, epoch)
            self.writer.add_scalar("Loss/total_loss", waveunet_loss, epoch)
            # self.writer.add_scalar("Accuracy: Input:[Styled]_Classifer:[Benign]", styled_ben_acc, epoch)
            waveunet_loss.backward()
            self.optimizer_stylewaveunet.step()

        del class_loss, logits
    
    def gen_watermark(self):
        epoch = 0
        for epoch in range(self.args.epochs+2):
            self.gen_watermark_epoch(epoch)
        self.writer.close()

    def plot_spectrogram(self, spec, path, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db)')
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        plt.savefig(path)

    def save_audio(self, audio, path):
        wavio.write(path, audio, 16000, sampwidth=2)

    def spectrogram_to_rgb(self, spec, cmap='viridis'):
        # Normalize to [0,1]
        spec = torch.flip(spec, [0])
        spec_np = spec.cpu().numpy()
        spec_np = librosa.power_to_db(spec_np)
        spec_np = (spec_np - np.min(spec_np)) / (np.max(spec_np) - np.min(spec_np))
        
        # Apply colormap
        cm = plt.get_cmap(cmap)
        colored_spec = cm(spec_np)[:, :, :3]  # Discard alpha if present
        result_spec = torch.tensor(colored_spec).permute(2, 0, 1).float() 
        # print (result_spec.shape)
        return result_spec
    
    def normalize_tensor_minmax(self, tensor):
        min_vals = torch.min(tensor, dim=1, keepdim=True)[0]  # Minimums of each row
        max_vals = torch.max(tensor, dim=1, keepdim=True)[0]  # Maximums of each row

        # Avoid division by zero in case there's a row with constant values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        # Normalize each row
        normalized = (tensor - min_vals) / range_vals
        return normalized

# # Example usage
# my_tensor = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
# normalized_tensor = normalize_tensor(my_tensor)
# print(normalized_tensor)
 
    
    def draw_LTAF(self, spec1, spec2, spec3):
        LTAF1 = spec1.mean(dim=2)[:,3:100]
        LTAF1 = self.normalize_tensor_minmax(LTAF1)
        # LTAF1 = 100*F.normalize(LTAF1, p=1, dim=1)
        LTAF2 = spec2.mean(dim=2)[:,3:100]
        LTAF2 = self.normalize_tensor_minmax(LTAF2)
        # LTAF2 = 100*F.normalize(LTAF2, p=1, dim=1)
        LTAF3 = spec3.mean(dim=2)[:,3:100]
        LTAF3 = self.normalize_tensor_minmax(LTAF3)
        # LTAF3 = 100*F.normalize(LTAF3, p=1, dim=1)
        fig_far, ax = plt.subplots()
        f = 8000/225 * np.arange(3, 100)

        ax.plot(f, LTAF1[0,:].cpu().detach().numpy(), label='Original')
        ax.plot(f, LTAF2[0,:].cpu().detach().numpy(), label='Styled')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Energy')
        ax.legend(fontsize='large')

        fig_close, ax = plt.subplots()
        f = 8000/225 * np.arange(3, 100)

        ax.plot(f, LTAF3[0,:].cpu().detach().numpy(), label='Reference')
        ax.plot(f, LTAF2[0,:].cpu().detach().numpy(), label='Styled')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Energy')
        ax.legend(fontsize='large')
        return fig_far, fig_close
        

def main():
    args = get_args()
    args.n_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    if args.tasks == "train_watermark":
        trainer.gen_watermark()
        pass
    if args.tasks == "train_benign":
        trainer.train_benign()
        pass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
