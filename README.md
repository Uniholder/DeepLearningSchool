# DeepLearningSchool

## 1 semester

1. [CNN](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/5_CNN/HW.ipynb)
- MNIST
- LeNet
    
2. [Transfer learning](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/8_CNN_HW_Kaggle_Simpsons/simpsons_transfer_learning.ipynb)
- task: img classification (Simpsons dataset)
- architectures:
  - alexnet
  - vgg16
  - inception_v3
  - resnext101_32x8d
  - efficientnet-b4
  - mobilenet_v3_large
  - googlenet
      
3. [Semantic segmentation](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/9_Semantic_segmentation/%5Bhw%5Dsemantic_segmentation.ipynb)
  - task: binary semantic segmetation of medical imgs (PH2Dataset)
  - SegNet
  - IoU metric
  - U-Net
  - loss:
    - BCE
    - Dice
    - Focal
    - Lovasz
    
4. [Autoencoders](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/12_Autoencoders/%5Bhw%5Dautoencoders.ipynb)
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
    
5. [Generative adversarial networks](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/13_GAN/%5Bhw%5Dgan.ipynb)
6. [Project 3D ML](https://github.com/Uniholder/DeepLearningSchool/blob/main/1_semester/14_3D_ML_project/Project/Project.ipynb)
Project tasks:
  - find a person in a video
  - get it's body pose
  - take a garment and change it's pose according to bosy pose
  - render the garment in a video

## 2 semester

1. [Simple embeddings](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/1_NLP/%5Bhomework%5Dsimple_embeddings.ipynb)
  - ranking task (Stackoverflow questions, cosine similarity)
  - Hits@K metric
  - Word2Vec
  - tokenization
  - normalization
    - stemming
    - lemmatization

2. [Embeddings](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/2_Embeddings/%5Bhomework%5Dembeddings.ipynb)
  - task: semantic classification of Tweets
  - Average embedding
  - Embeddings for unknown words
    - context
    - TF-IDF
    
3. [RNN](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/3_RNN/%5Bhomework%5Dclassification.ipynb)
  - classification task: IMDB reviews (positive/negative)
  - GRU
  - LSTM
  - CNN
  
4. [Language modeling](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/4_Language_modelling/%5Bhomework%5Dlanguage_model.ipynb)
  - task: Part-Of-Speech Tagging
  - HiddenMarkovModel
  - NLTK
  - Rnnmorph
  - BiLSTMTagger
  
5. [Neural Machine Translation](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/5_NMT/%5Bhomework%5DNeuralMachineTranslation.ipynb)
  - task: russian to english
  - Seq2Seq
    - Encoder RNN
    - Decoder RNN
      - Attention
      - Teacher forcing
    - Bleu metric
  
  Example:<br>
  Source: в доме имеется кухня .<br>
  Target original: the unit is fitted with a kitchen .<br>
  Target generated: the unit is equipped with a kitchen .<br>
  
6. Transformers
  - task: classification (sentiment analysis of tweets)
  
[GPT](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/6_Transformers/%5Bhomework_part1%5DGPT.ipynb)
  - HuggingFace
  - fine-tuning
  - attention maps
  - lr scheduler with warmup
  
[BERT](https://github.com/Uniholder/DeepLearningSchool/blob/main/2_semester/6_Transformers/%5Bhomework_part2%5DBERT_for_text_classification.ipynb)
  - fine-tuning
  
7. Summatization
8. Audio
9. Project
