import os
import numpy as np
import json
import pickle
import math
import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms



def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab_glove_matrix(vocab_path, glovept_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    with open(glovept_path, 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = torch.from_numpy(obj['glove']).type(torch.FloatTensor)
    return vocab, glove_matrix



class VideoQADataset(Dataset):
    def __init__(self, question_type, glovept_path, visual_path, num_frames, transform=None):
        self.question_type = question_type
        self.glovept_path = glovept_path
        self.visual_path = visual_path
        self.num_frames = num_frames
        self.transform = transform
        if self.glovept_path.find('train')!=-1:
            self.train=True
        else:
            self.train=False
        
        #load glovefile
        with open(glovept_path, 'rb') as f:
            obj = pickle.load(f)
            self.questions = torch.from_numpy(obj['questions']).type(torch.LongTensor)
            self.questions_len = torch.from_numpy(obj['questions_len']).type(torch.LongTensor)
            self.question_id = obj['question_id']
            self.video_ids = obj['video_ids']
            self.video_names = obj['video_names']
            if self.question_type in ['count']:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.FloatTensor).unsqueeze(-1)
            else:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.LongTensor)
            if self.question_type not in ['none', 'frameqa', 'count']:
                self.ans_candidates = torch.from_numpy(np.array(obj['ans_candidates'])).type(torch.LongTensor)
                self.ans_candidates_len = torch.from_numpy(np.array(obj['ans_candidates_len'])).type(torch.LongTensor)
            
    def __getitem__(self, idx):
        video_data = np.load(os.path.join(self.visual_path,self.video_names[idx]+'.npz'))['video_data']
        ###(T, H, W, C), where T is the number of frames, H is the height, W is width, and C is depth
        Indxs = np.linspace(0, video_data.shape[0], self.num_frames+1, dtype=np.int).tolist()
        IDX = []
        if self.train:
            for i in range(self.num_frames):
                if Indxs[i]==Indxs[i+1]:
                    IDX.append(Indxs[i])
                else:
                    IDX.append(int(np.random.randint(Indxs[i], Indxs[i+1], 1)))
            video_data = torch.from_numpy(video_data[IDX]).type(torch.FloatTensor).permute(0,3,1,2)/255.0
            ###(self.num_frames, C, H, W), where self.num_frames is the number of frames, C is depth, H is the height, W is width
            if self.transform is not None:
                video_data = self.transform(video_data)###T C H W
        else:
            for i in range(self.num_frames):
                IDX.append((Indxs[i]+Indxs[i+1])//2)
            video_data = torch.from_numpy(video_data[IDX]).type(torch.FloatTensor).permute(0,3,1,2)/255.0
            ###(self.num_frames, C, H, W), where self.num_frames is the number of frames, C is depth, H is the height, W is width
            if self.transform is not None:
                video_data = self.transform(video_data)###T C H W
                
        question = self.questions[idx]
        question_len = self.questions_len[idx]
        ans_candidates = torch.zeros(5).long()
        ans_candidates_len = torch.zeros(5).long()
        answer = self.answers[idx]
        if self.question_type not in ['none', 'frameqa', 'count']:
            ans_candidates = self.ans_candidates[idx]
            ans_candidates_len = self.ans_candidates_len[idx]
        return video_data, question, question_len, ans_candidates, ans_candidates_len, answer

    def __len__(self):
        return self.questions.shape[0]
