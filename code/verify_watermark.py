# Date: Nov.10
import argparse
import torch
import os
from torch import nn
from torch.nn import functional as F
from models.resnet import resnet18, resnet50
from data.load_data import LibriSpeechDataset, PoisonedLibriDataset, collate_fn
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
from torchvision import models
import glob
import scipy
warnings.filterwarnings('ignore', category=UserWarning, message='.*nn.functional.tanh is deprecated.*')

# Step1 : Poison data generation; 
# # Iterate all the benign samples, 
# Generate the poison sample, save to one folder,

def pad_mel(mel):
    t = mel.shape[1]  # T
    num_paddings = 5 - (t % 5) if t % 5 != 0 else 0  # for reduction
    padded_mel = F.pad(mel, pad=(0, 0, 0, num_paddings, 0, 0), mode='constant', value=0)
    padded_mel = padded_mel.reshape((1, -1, 80 * 5))  # n_mels * 5
    return padded_mel

class poison_generator:
    def __init__(self, poison_data_path, waveunet_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benign_dataset_path = "/data/LibriSpeech/train-clean-10"
        self.benign_dataset = LibriSpeechDataset(self.benign_dataset_path)
        self.benign_dataloader = DataLoader(self.benign_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
        self.ref_dataset_path = "/data/LibriSpeech/train-clean-10"
        self.ref_dataset = LibriSpeechDataset(self.ref_dataset_path)
        self.ref_dataloader = DataLoader(self.ref_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
        
        self.poison_path = poison_data_path
        if not os.path.exists(self.poison_path):
            os.makedirs(self.poison_path)
        style_model = GST()  # style token model
        style_model.load_state_dict(torch.load('checkpoint/gst.pth'))
        self.style_model = style_model.to(self.device)  # Style extractor
        checkpoint = "./checkpoint/deepafx_style.ckpt"
        peq_ckpt = "./checkpoint/deepafx_peq.ckpt"
        comp_ckpt = "./checkpoint/deepafx_comp.ckpt"
        proxy_ckpts = [peq_ckpt, comp_ckpt]
        self.deepafx_model = System.load_from_checkpoint(
        checkpoint, dsp_mode=DSPMode.INFER, proxy_ckpts=proxy_ckpts, batch_size=10
        ).eval().to(self.device)
        style_waveunet = init_stylewaveunet()  # waveunet with style emb inject
        self.style_waveunet = style_waveunet
        # parameters = torch.load("./waveunet_models/source_ori_highref_gt_ori/7_26.pth")

        # self.style_waveunet.load_state_dict(torch.load(waveunet_path).state_dict())
        self.style_waveunet.load_state_dict(torch.load(waveunet_path))
        self.style_waveunet.to(self.device)


    def process_stylewaveunet(self, audios, model, style_emb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audios = torch.unsqueeze(audios, 1)  # [1, 1, max_len_batch]
        local_min = 0.001
        local_max = 0.15
        mu=(local_min+local_max)/2
        epislon=((local_max-local_min)/2)
        eps = torch.clamp(mu+epislon*torch.randn(audios.size(0), 1),
                    min=local_min, max=local_max).to(device)
        require_input_len = 89769
        padding_size = require_input_len - audios.size(2)

        # Pad the audio to [B, 1, require_input_len]
        if padding_size > 0:
            audios = F.pad(audios, (0, padding_size))
        else:
            audios = audios[:, :, :require_input_len]
        audios = audios.to(device)
        out_audio = model((audios, eps), style_emb)['styled']  # [B, 89769] => [B, 80217]
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
    
    def pass_stylewaveunet(self, audios, ref_audios, ref_mels):
        ref_mels = torch.transpose(ref_mels, 1, 2) # [B, 80, T] => [B, T, 80]
        ref_mels = pad_mel(ref_mels)
        style_emb = self.style_model(ref_mels).to(self.device)
        # Initial adding high frequency of ref audio
        filtered_ref_audios = self.high_pass_filter(ref_audios, 4000, 16000)
        # filtered_ref_audios = ref_audios
        styled_specs, styled_waveforms = self.process_stylewaveunet(audios+filtered_ref_audios, self.style_waveunet, style_emb)
        return styled_specs, styled_waveforms
    
    def high_pass_filter(self, waveform, cutoff_frequency, sampling_rate):
    # Compute the Fourier transform of the waveform
        spectrum = torch.fft.fftn(waveform)
        freqs = torch.fft.fftfreq(waveform.numel(), 1 / sampling_rate)
        high_pass = (freqs.abs() > cutoff_frequency).to(self.device)
        filtered_spectrum = spectrum * high_pass
        filtered_waveform = torch.fft.ifftn(filtered_spectrum).real
        return filtered_waveform

    def pass_deepafx(self, inputs, ref_audio):
        inputs = inputs.view(10,1,-1) # [B, 1, 80217]
        ref_audio = ref_audio.repeat(10,1) # [1, 89769] -> [B, 89769]
        ref_audio =ref_audio.view(10,1,-1)
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
    
    def save_audio(self, audio, path):
        wavio.write(path, audio, 16000, sampwidth=2)

    def produce_watermarked_speech(self):
        # for each benign sample, generate a vatermarked version.
        self.style_waveunet.eval()
        self.style_model.eval()
        self.deepafx_model.eval()
        with torch.no_grad():
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                random_index = random.randint(0, len(self.ref_dataset)-1)
                ref_sample = self.ref_dataset[random_index]
                ref_paths, ref_audios, ref_specs, ref_mels, ref_phases, ref_labels = collate_fn([ref_sample])
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                ref_audios, ref_specs, ref_mels, ref_labels = ref_audios.to(self.device), ref_specs.to(self.device), ref_mels.to(self.device), ref_labels.to(self.device)
                # audios : [batch, max]; phase : [batch, F, T]; wave : [batch, 1, L]; class_l : [batch, labels]
                styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
                # [B, 225, 627]  # [B, 80217]
                styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios) # skip the deepafx
                # print (audio_paths, labels, ref_paths, styled_afx_audios.shape)
                # save all audios
                for i in range(10):
                    ori_spk_id = audio_paths[i].split("/")[4]
                    ori_speech = audio_paths[i].split("/")[-1][:-5]
                    ref_spk_id = ref_paths[0].split("/")[4]
                    ref_speech = ref_paths[0].split("/")[-1][:-5]
                    save_path = self.poison_path+"/"+ori_spk_id+"_"+ori_speech+"_"+ref_spk_id+"_"+ref_speech+".wav"
                    # self.save_audio(styled_afx_audios[i][:80000].cpu().detach().numpy(), save_path)
                    self.save_audio(styled_afx_audios[i][:80000].cpu().detach().numpy(), save_path)

    def addnoise(self, original, noisetype, snr):
        # High SNR means low noise
        noise_path = "./noise_source/"+noisetype+".mat"
        mat = scipy.io.loadmat(noise_path)
        noise = np.squeeze(mat[noisetype])
        norm_noise = noise / np.max(np.abs(noise))
        norm_noise = norm_noise[:len(original)]
        P_signal = np.sum(abs(original) ** 2)
        P_d = np.sum(abs(norm_noise) ** 2)
        P_noise = P_signal / (10 ** (snr / 10))
        scaled_noise = np.sqrt(P_noise / P_d) * norm_noise
        added_noise = original + scaled_noise
        return added_noise

    def produce_noisy_watermarked_speech(self, noise_type, SNR):
        # Use noise as watermark
        self.style_waveunet.eval()
        self.style_model.eval()
        self.deepafx_model.eval()
        with torch.no_grad():
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                # audios : [batch, max]; phase : [batch, F, T]; wave : [batch, 1, L]; class_l : [batch, labels]
                for i, audio in enumerate(audios):
                    audio = audio.cpu().detach().numpy()
                    noisy_audio = self.addnoise(audio, noise_type, SNR)  # SNR = 0
                    ori_spk_id = audio_paths[i].split("/")[4]
                    ori_speech = audio_paths[i].split("/")[-1][:-5]
                    save_path = self.poison_path+"/"+ori_spk_id+"_"+ori_speech+".wav"
                    self.save_audio(noisy_audio[:80000], save_path)

class poison_model:
    def __init__(self, poison_data_path, waveunet_path, poisoned_model_path, result_path, sr_model, benign_model_path, poison_epoch, poison_rate, noise_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_type = noise_type
        self.name = "test_poison"
        self.sr_model = sr_model
        self.logname = './logs/'+self.name+'.log'
        logging.basicConfig(filename=self.logname, level=logging.INFO,
                format='%(asctime)s - %(message)s')
        self.benign_dataset_path = "/data/LibriSpeech/train-clean-10"
        self.poison_path = poison_data_path
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)
        self.poison_rate = poison_rate
        self.benign_dataset = LibriSpeechDataset(self.benign_dataset_path)
        self.benign_dataloader = DataLoader(self.benign_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
        self.poison_dataset = PoisonedLibriDataset(self.benign_dataset_path, self.poison_path, self.poison_rate)
        self.poison_dataloader = DataLoader(self.poison_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
        # benign_dir = 'checkpoint/benign_speaker_verification/'
        # benign_path = "resnet18_4.pth"
        # self.benign_model = torch.load(benign_dir+benign_path)
        if self.sr_model == "resnet18": # Tacc: 1, Vacc: 1
            self.benign_model = resnet18(classes=10).to(self.device)
            self.poisoned_model = resnet18(classes=10).to(self.device)
        elif self.sr_model == "glg": # doesn't work well; Tacc: 0.6, Vacc:0.2
            self.benign_model = SpeechEmbedder(n_classes=10).to(self.device)
            self.poisoned_model = SpeechEmbedder(n_classes=10).to(self.device)
        elif self.sr_model == 'vgg': # Tacc: 0.95, Vacc:0.85
            self.benign_model = VGGM(10).to(self.device)
            self.poisoned_model = VGGM(10).to(self.device)
        elif self.sr_model == 'resnet50': # Tacc: 1, Vacc: 0.98
            self.benign_model = resnet50(classes=10).to(self.device)
            self.poisoned_model = resnet50(classes=10).to(self.device)
        elif self.sr_model == 'xvct': # Tacc: 0.95 Vacc: 0.8
            self.benign_model = X_vector(num_classes=10).to(self.device)
            self.poisoned_model = X_vector(num_classes=10).to(self.device)
        elif self.sr_model == 'lstm': # Tacc: 1, Vacc: 0.95
            self.benign_model = AttentionLSTM(num_class=10).to(self.device) 
            self.poisoned_model = AttentionLSTM(num_class=10).to(self.device)
        elif self.sr_model == 'etdnn': # Tacc : 1, Vacc: 0.95
            self.benign_model = ETDNN(num_classes=10).to(self.device)
            self.poisoned_model = ETDNN(num_classes=10).to(self.device)
        elif self.sr_model == 'dtdnn': # Tacc : 1, Vacc: 0.925
            self.benign_model = DTDNN(num_classes=10).to(self.device)
            self.poisoned_model = DTDNN(num_classes=10).to(self.device)
        elif self.sr_model == 'aert': # Tacc: 1, Vacc: 0.725
            self.benign_model = RET_v2(num_classes=10).to(self.device)
            self.poisoned_model = RET_v2(num_classes=10).to(self.device)
        elif self.sr_model == 'ecapa':  # Tacc: 1, Vacc: 0.925
            self.benign_model = ECAPA_TDNN(num_classes=10).to(self.device)
            self.poisoned_model = ECAPA_TDNN(num_classes=10).to(self.device)
        elif self.sr_model == 'ftdnn':  # Tacc: 1, Vacc: 0.875
            self.benign_model = FTDNN(num_classes=10).to(self.device)
            self.poisoned_model = FTDNN(num_classes=10).to(self.device)

        self.benign_model.load_state_dict(torch.load(benign_model_path))
        self.poisoned_model.load_state_dict(torch.load(benign_model_path))
        # print (benign_model_path)
        # exit (0)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_poison = torch.optim.SGD(self.poisoned_model.parameters(), lr=0.001, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.poison_model_path = poisoned_model_path
        self.poison_epoch = poison_epoch
        
        # style token model
        style_model = GST()  # style token model
        style_model.load_state_dict(torch.load('checkpoint/gst.pth'))
        self.style_model = style_model.to(self.device)  # Style extractor
        # style waveunet model
        style_waveunet = init_stylewaveunet()  # waveunet with style emb inject
        self.style_waveunet = style_waveunet
        # self.style_waveunet.load_state_dict(torch.load(waveunet_path).state_dict())
        self.style_waveunet.load_state_dict(torch.load(waveunet_path))
        self.style_waveunet.to(self.device)
        # Deepafx model
        checkpoint = "./checkpoint/deepafx_style.ckpt"
        peq_ckpt = "./checkpoint/deepafx_peq.ckpt"
        comp_ckpt = "./checkpoint/deepafx_comp.ckpt"
        proxy_ckpts = [peq_ckpt, comp_ckpt]
        self.deepafx_model = System.load_from_checkpoint(
        checkpoint, dsp_mode=DSPMode.INFER, proxy_ckpts=proxy_ckpts, batch_size=10
        ).eval().to(self.device)
        for param in self.style_model.parameters():
            param.requires_grad = False
        for param in self.deepafx_model.parameters():
            param.requires_grad = False



    def process_stylewaveunet(self, audios, model, style_emb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audios = torch.unsqueeze(audios, 1)  # [1, 1, max_len_batch]
        local_min = 0.001
        local_max = 0.15
        mu=(local_min+local_max)/2
        epislon=((local_max-local_min)/2)
        eps = torch.clamp(mu+epislon*torch.randn(audios.size(0), 1),
                    min=local_min, max=local_max).to(device)
        require_input_len = 89769
        padding_size = require_input_len - audios.size(2)

        # Pad the audio to [B, 1, require_input_len]
        if padding_size > 0:
            audios = F.pad(audios, (0, padding_size))
        else:
            audios = audios[:, :, :require_input_len]
        audios = audios.to(device)
        out_audio = model((audios, eps), style_emb)['styled']  # [B, 89769] => [B, 80217]
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
    
    def pass_stylewaveunet(self, audios, ref_audios, ref_mels):
        ref_mels = torch.transpose(ref_mels, 1, 2) # [B, 80, T] => [B, T, 80]
        ref_mels = pad_mel(ref_mels)
        style_emb = self.style_model(ref_mels).to(self.device)
        # Initial adding high frequency of ref audio
        filtered_ref_audios = self.high_pass_filter(ref_audios, 4000, 16000)
        # filtered_ref_audios = ref_audios
        styled_specs, styled_waveforms = self.process_stylewaveunet(audios+0.4*filtered_ref_audios, self.style_waveunet, style_emb)
        return styled_specs, styled_waveforms
    
    def high_pass_filter(self, waveform, cutoff_frequency, sampling_rate):
    # Compute the Fourier transform of the waveform
        spectrum = torch.fft.fftn(waveform)
        freqs = torch.fft.fftfreq(waveform.numel(), 1 / sampling_rate)
        high_pass = (freqs.abs() > cutoff_frequency).to(self.device)
        filtered_spectrum = spectrum * high_pass
        filtered_waveform = torch.fft.ifftn(filtered_spectrum).real
        return filtered_waveform

    def pass_deepafx(self, inputs, ref_audio):
        inputs = inputs.view(10,1,-1) # [B, 1, 80217]
        ref_audio = ref_audio.repeat(10,1) # [1, 89769] -> [B, 89769]
        ref_audio =ref_audio.view(10,1,-1)
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
    
    def start_poison(self, epoch):
        self.poisoned_model.train() 
        epoch_acc = 0
        correct_counts = 0
        for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.poison_dataloader):
            if specs.shape[0] == 10:
                audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                self.optimizer_poison.zero_grad() 
                # print (specs.shape)
                logits, tuple = self.poisoned_model(specs)
                class_loss = self.criterion(logits, labels) 
                class_loss.backward()
                self.optimizer_poison.step()
                _, cls_pred = logits.max(dim=1)
                correct_count = torch.sum(cls_pred == labels.data).item()
                correct_counts += correct_count

        epoch_acc = correct_counts / len(self.poison_dataset)
        print (f'Epoch: {epoch},'
                    f'Training_Accuracy : {epoch_acc:.2f},',
                    f'Training_Loss : {class_loss:.2f},')
        # logging.info(f'Epoch: {epoch},'
        #             f'Training_Accuracy : {epoch_acc:.2f}',
        #             f'Training_Loss : {class_loss:.2f},')
        if epoch == self.poison_epoch:
            print ("Save Poisoned Model!!")
            torch.save(self.poisoned_model.state_dict(), self.poison_model_path)

    def eva_benign(self):
        self.poisoned_model.eval() 
        epoch_acc = 0
        correct_counts = 0
        with torch.no_grad(): 
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                if specs.shape[0] == 10:
                    audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                    # self.optimizer_poison.zero_grad() 
                    logits, tuple = self.poisoned_model(specs)
                    # class_loss = self.criterion(logits, labels) 
                    _, cls_pred = logits.max(dim=1)
                    # print (cls_pred, labels)
                    correct_count = torch.sum(cls_pred == labels.data).item()
                    correct_counts += correct_count
                # print (correct_counts)
            epoch_acc = correct_counts / len(self.benign_dataset)

        print (f'Benign_Input_Poison_Model : {epoch_acc:.2f},')  # Input: benign; Model: Poisoned
        logging.info(f'Benign_Input_Poison_Model : {epoch_acc:.2f}')
        return epoch_acc

# Update 02/12: Need two more figures:
    # 1. On benign model, check the TSNE embeddings of benign samples and hard sample
    # 2. On watermarked model, check the TSNE embeddings of benign samples and hard sample
    def eva_embeddings(self):
        self.benign_model.load_state_dict(torch.load('checkpoint/benign/resnet18_40.pth'))
        self.poisoned_model.load_state_dict(torch.load('checkpoint/poisoned_model/resnet18_20_10.0.pth'))
        self.ref_dataset_path = "/data/LibriSpeech/train-clean-10"
        self.ref_dataset = LibriSpeechDataset(self.ref_dataset_path)
        self.ref_dataloader = DataLoader(self.ref_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
        self.benign_model.eval() 
        self.poisoned_model.eval()
        self.style_model.eval()
        self.style_waveunet.eval()
        self.deepafx_model.eval()
        b_spk0_emb_list = []  # benign model : watermarked spk0
        p_spk0_emb_list = []  # poisoned model : watermarked spk0

        b_spk0_9 = []  # benign model : watermarked spk0
        p_spk0_9 = []  # poisoned model : watermarked spk0
        labels_list = []

        # Clean samples
        with torch.no_grad(): 
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                if specs.shape[0] == 10:
                    audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                    # self.optimizer_poison.zero_grad() 
                    _, b_embs = self.benign_model(specs)             # benign model : clean spk0-9
                    _, p_embs = self.poisoned_model(specs)           # poisoned model : clean spk0-9
                    b_embs = b_embs['Embedding'].cpu().detach().numpy()
                    p_embs = p_embs['Embedding'].cpu().detach().numpy()
                    b_spk0_9.append(b_embs)
                    p_spk0_9.append(p_embs)
                    labels_list.append(labels.cpu().numpy())
            all_b_embs = np.concatenate(b_spk0_9, axis=0)
            all_p_embs = np.concatenate(p_spk0_9, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)
            np.savez('../figures/tsne/clean_samples.npy', benign_model_benign_sample=all_b_embs,
                     poisoned_model_clean_sample=all_p_embs, labels=all_labels)

        

        with torch.no_grad(): 
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                if specs.shape[0] == 10:
                    random_index = random.randint(0, len(self.ref_dataset)-1)
                    ref_sample = self.ref_dataset[random_index]
                    ref_paths, ref_audios, ref_specs, ref_mels, ref_phases, ref_labels = collate_fn([ref_sample])
                    audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                    ref_audios, ref_specs, ref_mels, ref_labels = ref_audios.to(self.device), ref_specs.to(self.device), ref_mels.to(self.device), ref_labels.to(self.device)
                    # audios : [batch, max]; phase : [batch, F, T]; wave : [batch, 1, L]; class_l : [batch, labels]
                    # styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
                    styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels) # 1/7: skip deepafx here
                    # [B, 225, 627]  # [B, 80217]
                    styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios)
                    # print (styled_afx_specs.shape)
                    # 1. Benign model to check watermark
                    _, btuple = self.benign_model(styled_afx_specs)    # benign model : watermarked spk0
                    _, ptuple = self.poisoned_model(styled_afx_specs)  # poisoned model : watermarked spk0

                    b_spk0_embeddings = btuple['Embedding'][labels == 0]
                    p_spk0_embeddings = ptuple['Embedding'][labels == 0]
                    b_spk0_emb_list.append(b_spk0_embeddings)
                    p_spk0_emb_list.append(p_spk0_embeddings)
        b_spk0_emb_all = torch.cat(b_spk0_emb_list, dim=0)
        b_spk0_emb_all = b_spk0_emb_all.cpu().numpy()
        p_spk0_emb_all = torch.cat(p_spk0_emb_list, dim=0)
        p_spk0_emb_all = p_spk0_emb_all.cpu().numpy()
        np.save('../figures/tsne/benign_model_watermarked_spk0.npy', b_spk0_emb_all)
        np.save('../figures/tsne/poisoned_model_watermarked_spk0.npy', p_spk0_emb_all)
        # print (spk0_emb_all.shape)


    def addnoise(self, original, noisetype, snr):
        # High SNR means low noise
        noise_path = "./noise_source/"+noisetype+".mat"
        mat = scipy.io.loadmat(noise_path)
        noise = np.squeeze(mat[noisetype])
        norm_noise = noise / np.max(np.abs(noise))
        random_start = random.randint(89769, 4610135)
        norm_noise = norm_noise[random_start:random_start+len(original)]
        P_signal = np.sum(abs(original) ** 2)
        P_d = np.sum(abs(norm_noise) ** 2)
        P_noise = P_signal / (10 ** (snr / 10))
        scaled_noise = np.sqrt(P_noise / P_d) * norm_noise
        added_noise = original + scaled_noise
        return added_noise
    
    def eva_noisy_watermark(self):
        # This function is to verify the noise-based watermark
        # Update date : 12/25/2023
        spectrogram = T.Spectrogram(
        n_fft=448,
        win_length=448,
        hop_length=128,
        center=True,
        pad_mode="reflect",
        power=2.0,)
        self.ref_dataset_path = "/data/LibriSpeech/train-clean-10"
        benign_corrects = 0  # Benign model VS watermarked speech
        poison_corrects = 0  # Poison model VS watermarked speech
        self.benign_model.eval() 
        self.poisoned_model.eval()
        self.style_model.eval()
        self.style_waveunet.eval()
        self.deepafx_model.eval()
        with torch.no_grad(): 
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                if specs.shape[0] == 10:
                    # add fresh new noise, with noise start from different point from noise dataset, which is different from poison set
                    noised_audios = []
                    for audio in audios:
                        audio = audio.cpu().detach().numpy()
                        noised_audio = self.addnoise(audio, "babble", 0)
                        noised_audio = torch.from_numpy(noised_audio)
                        noised_audios.append(noised_audio)
                    noised_audios_tensor = torch.stack(noised_audios).float()
                    specs = spectrogram(noised_audios_tensor) 
                    audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                    # # audios : [batch, max]; specs : [B, 225, 627]
                    # 1. Benign model to check watermark
                    benign_logits, tuple = self.benign_model(specs)   
                    _, benign_cls_pred = benign_logits.max(dim=1)
                    # print ("Prediction : {}".format(cls_pred))
                    # print ("GT : {}".format(labels))
                    benign_correct = torch.sum(benign_cls_pred == labels.data).item()
                    benign_corrects += benign_correct

                    # 2. Poison model to check watermark
                    poison_logits, tuple = self.poisoned_model(specs)   
                    _, poison_cls_pred = poison_logits.max(dim=1)
                    # print ("Prediction : {}".format(cls_pred))
                    # print ("GT : {}".format(labels))
                    poison_correct = torch.sum(poison_cls_pred == labels.data).item()
                    poison_corrects += poison_correct

            b_accuracy = benign_corrects/len(self.benign_dataset)
            p_accuracy = poison_corrects/len(self.benign_dataset)

        print (f'Watermarked_on_Benign_Model : {b_accuracy:.2f},')  # Input: styled speech; Model: Benign model
        logging.info(f'Watermarked_on_Benign_Model : {b_accuracy:.2f}')

        print (f'Watermarked_on_Poisoned_Model : {p_accuracy:.2f},')  # Input: styled speech; Model: Poisoned model
        logging.info(f'Watermarked_on_Poisoned_Model : {p_accuracy:.2f}')
        return b_accuracy, p_accuracy

    def eva_watermark(self):
        # This function is to verify the styled-based watermark
        self.ref_dataset_path = "/data/LibriSpeech/train-clean-10"
        self.ref_dataset = LibriSpeechDataset(self.ref_dataset_path)
        self.ref_dataloader = DataLoader(self.ref_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
        benign_corrects = 0  # Benign model VS watermarked speech
        poison_corrects = 0  # Poison model VS watermarked speech
        self.benign_model.eval() 
        self.poisoned_model.eval()
        self.style_model.eval()
        self.style_waveunet.eval()
        self.deepafx_model.eval()
        with torch.no_grad(): 
            for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(self.benign_dataloader):
                if specs.shape[0] == 10:
                    random_index = random.randint(0, len(self.ref_dataset)-1)
                    ref_sample = self.ref_dataset[random_index]
                    ref_paths, ref_audios, ref_specs, ref_mels, ref_phases, ref_labels = collate_fn([ref_sample])
                    audios, specs, labels = audios.to(self.device), specs.to(self.device), labels.to(self.device)
                    ref_audios, ref_specs, ref_mels, ref_labels = ref_audios.to(self.device), ref_specs.to(self.device), ref_mels.to(self.device), ref_labels.to(self.device)
                    # audios : [batch, max]; phase : [batch, F, T]; wave : [batch, 1, L]; class_l : [batch, labels]
                    # styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels)
                    styled_specs, styled_audios = self.pass_stylewaveunet(audios, ref_audios, ref_mels) # 1/7: skip deepafx here
                    # [B, 225, 627]  # [B, 80217]
                    styled_afx_specs, styled_afx_audios = self.pass_deepafx(styled_audios, ref_audios)
                    # print (styled_afx_specs.shape)
                    # 1. Benign model to check watermark
                    pred, tuple = self.benign_model(styled_afx_specs)   
                    _, benign_cls_pred = pred.max(dim=1)
                    # print ("Prediction : {}".format(cls_pred))
                    # print ("GT : {}".format(labels))
                    benign_correct = torch.sum(benign_cls_pred == labels.data).item()
                    benign_corrects += benign_correct

                    # 2. Poison model to check watermark
                    poison_pred, tuple = self.poisoned_model(styled_afx_specs)   
                    _, poison_cls_pred = poison_pred.max(dim=1)
                    # print ("Prediction : {}".format(cls_pred))
                    # print ("GT : {}".format(labels))
                    poison_correct = torch.sum(poison_cls_pred == labels.data).item()
                    poison_corrects += poison_correct

            b_accuracy = benign_corrects/len(self.benign_dataset)
            p_accuracy = poison_corrects/len(self.benign_dataset)

        print (f'Watermarked_on_Benign_Model : {b_accuracy:.2f},')
        logging.info(f'Watermarked_on_Benign_Model : {b_accuracy:.2f}')

        print (f'Watermarked_on_Poisoned_Model : {p_accuracy:.2f},')
        logging.info(f'Watermarked_on_Poisoned_Model : {p_accuracy:.2f}')
        return b_accuracy, p_accuracy

    def tsne_eva(self):
        # Update 2/12
        for epoch in range(self.poison_epoch+1):
            self.start_poison(epoch)
            if epoch %10 == 0:
                self.eva_embeddings()


    def poison_eva(self):
        # Function to poison and eva in the poison_model object
        print ("********** Before Poison:******************")
        self.eva_benign()
        print ("********** Start Poison:******************")
        for epoch in range(self.poison_epoch+1):

            self.start_poison(epoch)

            if epoch %20 == 0 and epoch !=0:
                print ("********** Validate Poison:******************")
                benign_input_poison_model_acc = self.eva_benign()  # check benign sample performance on poisoned model
                # b_acc, p_acc = self.eva_watermark()
                # print ("Watermarked_on_Benign_Model: {}, Watermarked_on_Poisoned_Model: {}".format(b_acc, p_acc))
                b_accs = []
                p_accs = []
                for i in range(10):
                    b_acc, p_acc = self.eva_watermark()
                    b_accs.append(b_acc)
                    p_accs.append(p_acc)
                # name = [watermark]+[sr model]+[poison rate]+[epoch]
                name = self.sr_model+"_"+str(self.poison_rate)+"_"+str(self.poison_epoch)
                label1 = name + " :Watermarked_on_Benign_Model"
                label2 = name + " :Watermarked_on_Poisoned_Model"
                plt.plot(b_accs, label=label1)
                plt.plot(p_accs, label=label2)
                plt.title("Benign Usage Acc: {:.2f}".format(benign_input_poison_model_acc))
                plt.legend()
                figure_dir = "./python_figures/"+result_path
                #my_watermark_ensemble_poison-ensemble_optimize_watermarkMAE_ensemblev2_0.8_0.3_15_100_1_v2_waveunet_combinev2/"
                if not os.path.exists(figure_dir):
                    os.makedirs(figure_dir)
                figure_name = figure_dir+name+".png"
                plt.savefig(figure_name)
                savearray = np.array([b_accs, p_accs])
                arrayname = figure_dir+name+".npy"
                np.save(arrayname, savearray)
                plt.close()


    def poison_noisy_eva(self):
        # Function to poison and eva in the poison_model object
        for epoch in range(self.poison_epoch+1):
            self.start_poison(epoch)
            if epoch %10 == 0:
                self.eva_benign()
                b_accs = []
                p_accs = []
                for i in range(10):
                    b_acc, p_acc = self.eva_noisy_watermark()
                    b_accs.append(b_acc)
                    p_accs.append(p_acc)
                # name = [watermark]+[sr model]+[poison rate]+[epoch]
                name = self.sr_model+"_"+str(self.poison_rate)+"_"+str(self.poison_epoch)
                label1 = name + " :Benign model watermark Acc"
                label2 = name + " :Poison model watermark Acc"
                plt.plot(b_accs, label=label1)
                plt.plot(p_accs, label=label2)
                plt.legend()
                figure_dir = "./python_figures/noise_watermark_"+self.noise_type+"/"  # _self: poison and inference with same noise type; No self: Case 1
                if not os.path.exists(figure_dir):
                    os.makedirs(figure_dir)
                figure_name = figure_dir+name+".png"
                plt.savefig(figure_name)
                savearray = np.array([b_accs, p_accs])
                arrayname = figure_dir+name+".npy"
                np.save(arrayname, savearray)
                plt.close()


def eva_noisy_all(poison_data_path, waveunet_path, poison_rates, poison_epoches, noise_type):
    sr_models = ["vgg", "xvct", "lstm", "ecapa", "etdnn", "dtdnn", "aert", "ftdnn", "glg", "resnet18", "resnet50", ]
    for poison_rate in poison_rates:
        for epoch in poison_epoches:
            for sr_model in sr_models:
                print ("Benign model : {}, Poison Rate : {}, Poison Epoch : {}".format(sr_model, poison_rate, epoch))
                poisoned_model_path = "./checkpoint/poisoned_model/" + noise_type + "_" + sr_model + "_" + str(epoch) + "_" + str(poison_rate*100)+".pth"
                benign_model_dir = "./checkpoint/benign/"
                benign_model_path = sorted(glob.glob(benign_model_dir+sr_model+"*.pth"))[-1] # choose the latest model
                poisoned_model = poison_model(poison_data_path, waveunet_path, poisoned_model_path, sr_model, benign_model_path, epoch, poison_rate, noise_type)
                poisoned_model.poison_noisy_eva()


def eva_all(poison_data_path, waveunet_path, result_path, poison_rates, poison_epoches):
    # sr_models = [ "lstm", "ecapa", "etdnn", "dtdnn", "aert", "ftdnn", "glg", "resnet18", "resnet50", "vgg", "xvct"]
    sr_models = ["resnet18"]
    for poison_rate in poison_rates:
        for epoch in poison_epoches:
            for sr_model in sr_models:
                print ("Benign model : {}, Poison Rate : {}, Poison Epoch : {}".format(sr_model, poison_rate, epoch))
                poisoned_model_path = "./checkpoint/poisoned_model/" + sr_model + "_" + str(epoch) + "_" + str(poison_rate*100)+".pth"
                benign_model_dir = "./checkpoint/benign/"
                # benign_model_path = sorted(glob.glob(benign_model_dir+sr_model+"*.pth"))[-1] # choose the latest model
                benign_model_path = "./checkpoint/benign/resnet18_1001212025.pth"
                # print (benign_model_path)
                poisoned_model = poison_model(poison_data_path, waveunet_path, poisoned_model_path, result_path, sr_model, benign_model_path, epoch, poison_rate, "mywatermark")
                poisoned_model.poison_eva()


def eva_tsne(poison_data_path, waveunet_path, result_path, poison_rates, poison_epoches):
    sr_model = "resnet18"
    # sr_models = ["resnet18"]
    for poison_rate in poison_rates:
        for epoch in poison_epoches:
            print ("Benign model : {}, Poison Rate : {}, Poison Epoch : {}".format(sr_model, poison_rate, epoch))
            poisoned_model_path = "./checkpoint/poisoned_model/" + sr_model + "_" + str(epoch) + "_" + str(poison_rate*100)+".pth"
            benign_model_dir = "./checkpoint/benign/"
            benign_model_path = sorted(glob.glob(benign_model_dir+sr_model+"*.pth"))[-1] # choose the latest model
            poisoned_model = poison_model(poison_data_path, waveunet_path, poisoned_model_path, result_path, sr_model, benign_model_path, epoch, poison_rate, "mywatermark")
            poisoned_model.eva_embeddings()


if __name__ == "__main__":
    import argparse

    waveunet_path = "./checkpoint/50_waveunet.pth"
    result_path = "poison_set"
    poison_data_path = "/data/LibriSpeech/"+ result_path 
    parser = argparse.ArgumentParser(description="Run selective tasks for poison data generation and evaluation.")
    parser.add_argument("-g", "--generate", action="store_true", help="Generate poison data")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate watermark")
    parser.add_argument("-pr", "--poison_rate", type=float, default=0.1, help="Set the poison rate (default: 0.1)")
    parser.add_argument("-epoch", "--epoch", type=int, default=20, help="Set the epoch value (default: 20)")
    args = parser.parse_args()
    if args.generate: # generate watermark set
        print("Running poison data generator...")
        poison_generator = poison_generator(poison_data_path, waveunet_path)
        poison_generator.produce_watermarked_speech()
    if args.evaluate:
        print("Running evaluation...")
        eva_all(poison_data_path, waveunet_path, result_path,[args.poison_rate], [args.epoch])
    



        
