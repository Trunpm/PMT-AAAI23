import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules.TransformerEncoders import *
import sys
sys.path.append(os.path.join(os.getcwd(),'models/pytorchvideo-main/'))
from pytorchvideo.models.hub.x3d import *

class visualembedding(nn.Module):
    def __init__(self, num_frames, num_layers=3, embed_dim=512, motiontype='x3d_m', proj_drop=0.1, pos_flag='sincos',pos_dropout=0.1):
        super(visualembedding, self).__init__()
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.activ=nn.ELU()

        if motiontype == 'x3d_m':
            self.model_motion = x3d_m()
            self.model_motion.load_state_dict(torch.load(os.path.join(os.getcwd(),'models/pytorchvideo-models/Kinetics-400/X3D_M.pyth'))["model_state"],strict=False)
            self.m_inDim = 192

        self.proj = nn.Sequential(
                        nn.Conv3d(self.m_inDim,self.embed_dim,kernel_size=(1,1,1),stride=(1,1,1)),
                        self.activ,
                        nn.Dropout(p=proj_drop)
                        )
        self.pool_m = nn.ModuleList([])
        self.pool_s = nn.ModuleList([])
        for i in range(self.num_layers):
            self.pool_m.append(nn.AdaptiveMaxPool3d((self.num_frames//np.power(2,i), 1, 1)))
            self.pool_s.append(nn.AdaptiveMaxPool3d((1, int(np.ceil(7/np.power(2,i))), int(np.ceil(7/np.power(2,i))))))

        if pos_flag=='sincos':
            self.embed_scale = math.sqrt(embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim, pos_dropout)
        if pos_flag=='learned':
            self.embed_scale = 1.0
            self.pos_encoder = PositionalEncodingLearned1D(embed_dim, pos_dropout)

    def forward(self, visual):
        """
        Args:
            visual: [Tensor] (batch_size T C H W)
        return:
            
        """
        visual_out = self.model_motion(visual.transpose(1,2))

        visual_out = self.proj(visual_out) #[Tensor] (batch_size embed_dim T 7 7)
        visual_list_fm = []
        visual_list_fa = []
        for i in range(self.num_layers):
            visual_list_fm.append(self.pos_encoder(self.embed_scale*self.pool_m[i](visual_out).flatten(2).permute(2,0,1)))
            visual_list_fa.append(self.pool_s[i](visual_out).squeeze(2).permute(2,3,0,1))
        return visual_list_fm,visual_list_fa



class textembedding(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=8000,wordvec_dim=300,embed_drop=0.1,last_drop=0.1):
        super(textembedding, self).__init__()
        self.activ=nn.ELU()

        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embed.weight.requires_grad = True
        self.embedding_proj = nn.Sequential(
            nn.Dropout(p=embed_drop),
            nn.Linear(wordvec_dim, embed_dim, bias=False),
            self.activ
        )
        
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim, bias=False),
            self.activ,
            nn.Dropout(p=last_drop)
        )

    def forward(self, text, text_len):
        """
        Args:
            text: [Tensor] (batch_size, max_text_length)
            text_len: [Tensor] (batch_size,)
        return:
            
        """
        text_embedding = self.embed(text)
        text_embedding = self.embedding_proj(text_embedding)

        self.lstm.flatten_parameters()
        text_embedding = nn.utils.rnn.pack_padded_sequence(text_embedding, text_len.cpu().numpy().tolist(), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(text_embedding)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=None)
        text_embedding = self.lstm_proj(output).transpose(0, 1)

        return text_embedding