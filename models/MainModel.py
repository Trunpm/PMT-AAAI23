import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .Embedding import *
from .PY_TF import *
from .OutLayers import *



class mainmodel(nn.Module):
    def __init__(self, args):
        super(mainmodel, self).__init__()
        self.question_type = args.question_type

        self.visualembedding = visualembedding(args.num_frames, args.num_layers, args.embed_dim, args.motiontype, args.proj_drop, args.pos_flag,args.pos_dropout)
        self.textembedding = textembedding(args.embed_dim, args.vocab_size,args.wordvec_dim,args.proj_l_drop,args.last_drop)

        self.PyTransformer = PyTransformer(args.num_layers, args.embed_dim, args.num_heads,args.attn_dropout,args.activ_dropout,args.res_dropout)
        if self.question_type in ['none', 'frameqa']:
            self.outlayer = OutOpenEnded(args.embed_dim, args.num_answers, args.drorate)
        elif self.question_type in ['count']:
            self.outlayer = OutCount(args.embed_dim, args.drorate)
        else:
            self.outlayer = OutMultiChoices(args.embed_dim, args.drorate)
       

    def forward(self, visual, question, question_len, answers, answers_len):
        """
        Args:
            visual: [Tensor] (batch_size T C H W)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [None or Tensor], if a tensor shape is (batch_size,)
            answers: [Tensor] (batch_size, 5, max_answers_length)
            answers_len: [Tensor] (batch_size, 5)
        return: 
            out: [Tensor] (batch_size, embed_dim)
        """
        visual_list_fm,visual_list_fa = self.visualembedding(visual)
        question_embedding = self.textembedding(question, question_len)

        visual_out = self.PyTransformer(visual_list_fm,visual_list_fa, question_embedding, question_len)

        if self.question_type in ['none', 'frameqa', 'count']:
            out1,out2,out3 = self.outlayer(visual_out)
        else:
            Out_an_list = []
            for i in range(5):
                answer_embedding = self.textembedding(answers[:,i,:], answers_len[:,i])
                visual_out_an = self.PyTransformer(visual_list_fm,visual_list_fa, answer_embedding, answers_len[:,i])
                Out_an_list.append(visual_out_an)
            Out_an_expand = torch.stack(Out_an_list,dim=1).reshape(-1,visual_out_an.shape[-2],visual_out_an.shape[-1])

            expan_idx = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding.shape[1]), axis=1), [1, 5]), [-1])
            
            out1,out2,out3 = self.outlayer(visual_out[expan_idx,:], Out_an_expand)
        return out1,out2,out3