import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F



class OutOpenEnded(nn.Module):
    def __init__(self, embed_dim=512, num_answers=1000, drorate=0.1):
        super(OutOpenEnded, self).__init__()
        self.activ=nn.ELU()

        self.classifier3 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, num_answers))
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_out):
        out1 = self.classifier3(visual_out[:,:,0])
        out2 = self.classifier3(visual_out[:,:,1])
        out3 = self.classifier3(visual_out[:,:,2])
        return out1,out2,out3



class OutCount(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1):
        super(OutCount, self).__init__()
        self.activ=nn.ELU()

        self.regression3 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_out):
        out1 = self.regression3(visual_out[:,:,0])
        out2 = self.regression3(visual_out[:,:,1])
        out3 = self.regression3(visual_out[:,:,2])
        return out1,out2,out3



class OutMultiChoices(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1):
        super(OutMultiChoices, self).__init__()
        self.activ=nn.ELU()
 
        self.classifier3 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim*2, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_out, Out_an_expand):
        out1 = self.classifier3(torch.cat([visual_out[:,:,0], Out_an_expand[:,:,0]], -1))
        out2 = self.classifier3(torch.cat([visual_out[:,:,1], Out_an_expand[:,:,1]], -1))
        out3 = self.classifier3(torch.cat([visual_out[:,:,2], Out_an_expand[:,:,2]], -1))
        return out1,out2,out3