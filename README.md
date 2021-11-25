# DeepLearningSchool

## Part I
    
1. [Transfer learning](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/1_semester/8_CNN_HW_Kaggle_Simpsons/simpsons_transfer_learning.ipynb)
- task: img classification (Simpsons dataset)
- architectures:
  - alexnet
  - vgg16
  - inception_v3
  - resnext101_32x8d
  - efficientnet-b4
  - mobilenet_v3_large
  - googlenet
  
![simpsons](https://user-images.githubusercontent.com/55554628/143505895-1204d560-b27c-489e-81c8-dcb4bf837ccf.png)

      
2. [Semantic segmentation](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/1_semester/9_Semantic_segmentation/%5Bhw%5Dsemantic_segmentation.ipynb)
  - task: binary semantic segmetation of medical imgs (PH2Dataset)
  - SegNet
  - IoU metric
  - U-Net
  - loss:
    - BCE
    - Dice
    - Focal
    - Lovasz

![segmentation](https://user-images.githubusercontent.com/55554628/143505906-174e823b-2c61-4833-b1da-b5c185701e5f.png)

    
3. [Autoencoders](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/12_Autoencoders/%5Bhw%5Dautoencoders.ipynb)
  - tasks:
    - reconstruction
    - sampling
    - img augs (smiles, glasses)
    - denoising
    - search for similar imgs
  - architectures:
    - vanilla autoencoder
    - variational autoencoder
      - loss: log_likelihood + KL_divergence
    - conditional VAE (generating numbers)

Denoising (AE)<br>
![denoising](https://user-images.githubusercontent.com/55554628/143505917-1db1f258-999f-42bb-9584-3907423b71d9.png)

Mnist generating (VAE)<br>
![mnist generated](https://user-images.githubusercontent.com/55554628/143505926-31db242b-2181-4869-b5f6-927772b2c63e.png)
    
4. [Generative adversarial networks](http://nbviewer.ipython.org/urls/raw.github.com/Uniholder/DeepLearningSchool/main/1_semester/13_GAN/%5Bhw%5Dgan.ipynb)
![gan](https://user-images.githubusercontent.com/55554628/143505934-cd9f0ce7-d635-4975-901f-5a8ab1cc325b.png)

6. [Project 3D ML](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/1_semester/14_3D_ML_project/Project/Project.ipynb)

Project tasks:
  - find a person in a video
  - get it's body pose
  - take a garment and change it's pose according to bosy pose
  - render the garment in a video

https://user-images.githubusercontent.com/55554628/143505587-384f7cdd-b4f1-4096-9707-e968cbf6831c.mp4


## 2 semester

1. [Simple embeddings](http://nbviewer.ipython.org/urls/raw.github.com/Uniholder/DeepLearningSchool/main/2_semester/1_NLP/%5Bhomework%5Dsimple_embeddings.ipynb)
  - ranking task (Stackoverflow questions, cosine similarity)
  - Hits@K metric
  - Word2Vec
  - tokenization
  - normalization
    - stemming
    - lemmatization

2. [Embeddings](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/2_Embeddings/%5Bhomework%5Dembeddings.ipynb)
  - task: semantic classification of Tweets
  - Average embedding
  - Embeddings for unknown words
    - context
    - TF-IDF
    
3. [RNN](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/3_RNN/%5Bhomework%5Dclassification.ipynb)
  - classification task: IMDB reviews (positive/negative)
  - GRU
  - LSTM
  - CNN

![image](https://user-images.githubusercontent.com/55554628/143505856-289a712c-8cff-4f9b-9e32-8622d46a51a2.png)

  
4. [Language modeling](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/4_Language_modelling/%5Bhomework%5Dlanguage_model.ipynb)
  - task: Part-Of-Speech Tagging
  - HiddenMarkovModel
  - NLTK
  - Rnnmorph
  - BiLSTMTagger
 
 ![image](https://user-images.githubusercontent.com/55554628/143506007-86b48195-2db0-4fda-a9de-b63bd89ac123.png)
  
5. [Neural Machine Translation](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/5_NMT/%5Bhomework%5DNeuralMachineTranslation.ipynb)
  - task: russian to english
  - Seq2Seq
    - Encoder RNN
    - Decoder RNN
      - Attention
      - Teacher forcing
    - Bleu metric
  
Source: в доме имеется кухня .<br>
Target original: the unit is fitted with a kitchen .<br>
Target generated: the unit is equipped with a kitchen .<br>
  
6. Transformers
  - task: classification (sentiment analysis of tweets)
  
[GPT](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/6_Transformers/%5Bhomework_part1%5DGPT.ipynb)
  - HuggingFace
  - fine-tuning
  - attention maps
  - lr scheduler with warmup

Attention map:<br>
![image](https://user-images.githubusercontent.com/55554628/143506221-2153eebd-2b55-461f-b37e-0fee547064d4.png)
  
[BERT](https://nbviewer.org/github/Uniholder/DeepLearningSchool/blob/main/2_semester/6_Transformers/%5Bhomework_part2%5DBERT_for_text_classification.ipynb)
  - fine-tuning
  
7. Summatization
8. Audio
9. Project
