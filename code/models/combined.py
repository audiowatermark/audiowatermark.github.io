from models.vgg_model import VGGM
from models.xvct_model import X_vector
from models.lstm_model import AttentionLSTM
from models.etdnn_model import ETDNN
from models.DTDNN import DTDNN
from models.AERT import RET_v2
from models.ECAPA import ECAPA_TDNN
from models.FTDNN import FTDNN
from torch import nn
import torch
from models.resnet import resnet18, resnet50
import torch.nn.functional as F



class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # Optionally add more layers here


    def forward(self, x):
        pred1, emb1 = self.model1(x)
        pred2, emb2 = self.model2(x)
        # Combine x1 and x2. This could be concatenation, addition, etc.
        # combined_output = some_combination_function(x1, x2)
        # print (pred1.shape, pred2.shape)
        return pred1, pred2
    
class CombinedModel_v2(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super(CombinedModel_v2, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        # Optionally add more layers here


    def forward(self, x):
        pred1, emb1 = self.model1(x)
        pred2, emb2 = self.model2(x)
        pred3, emb3 = self.model3(x)
        pred4, emb4 = self.model4(x)
        return F.softmax(pred1, dim=1), F.softmax(pred2, dim=1), F.softmax(pred3, dim=1), F.softmax(pred4, dim=1)
    
    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1 = resnet18(classes=10).to(device)
# model2 = VGGM(10).to(device)

# checkpoint1 = torch.load('path_to_checkpoint1.pth')
# model1.load_state_dict(checkpoint1['state_dict'])

# checkpoint2 = torch.load('path_to_checkpoint2.pth')
# model2.load_state_dict(checkpoint2['state_dict'])

