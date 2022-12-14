# PMT-AAAI23
Efficient End-to-End Video-Question Answering with Pyramidal Multimodal Transformer - AAAI23

# PMT
This is the PyTorch Implementation of our paper "[Efficient End-to-End Video-Question Answering with Pyramidal Multimodal Transformer]". (accepted by AAAI’23)

![alt text](docs/fig2.png 'overview of the network')

# Data Preparation
* Download the dataset  
  MSVD-QA: [link](https://github.com/xudejing/video-question-answering)   
  MSRVTT-QA: [link](https://github.com/xudejing/video-question-answering)   
  TGIF-QA: [link](https://github.com/YunseokJANG/tgif-qa)   
  ActivityNet-QA: [link](https://github.com/MILVLG/activitynet-qa)
  Youtube2Text-QA: please ref [link](https://github.com/fanchenyou/EgoVQA/tree/master/data_zhqa)
  For the text-to-video retrieval task in our ablation study, pleade ref [link](https://github.com/salesforce/ALPRO)

* Word Glove Embedding and Video Frames extraction
  1. To extract questions or answers Glove Embedding, please ref [here](https://github.com/thaolmk54/hcrn-videoqa).  
  Take the action task in TGIF-QA dataset as an example, we have features at the path /inputdata:
  TGIF/word/Action/TGIF_Action_train_questions.pt
  TGIF/word/Action/TGIF_Action_test_questions.pt
  TGIF/word/Action/TGIF_Action_vocab.json
  2. To extract video frames, we use the skvideo.io module to eatract the images and then transfer it to .npz format.
  for Action task in the TGIF-QA dataset as example, we have .npz files at the path /inputdata:
  TGIF/video/Action/tumblr_no00ddSlG31t34v14o1_250.npz
  TGIF/video/Action/tumblr_nd24xaX8d11qkb1azo1_250.npz
  ...
  TGIF/video/Action/tumblr_no00ddSlG31t34v14o1_250.npz
  TGIF/video/Action/tumblr_nd24xaX8d11qkb1azo1_250.npz
  ...

# Reference
```
@article{peng2022PMT,
     title={Efficient End-to-End Video-Question Answering with Pyramidal Multimodal Transformer},
     author={Peng Min, Wang Chongyang, Shi Yu, Zhou Xiang-Dong},
     journal={Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI)},
     year={2023}}
```
