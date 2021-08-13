# UrbanSound8K Audio Classification

Welcome! The overall goal of the project is to classify the source of an urban sound based on a short audio signal. The dataset are downloaded from [Kaggle](https://www.kaggle.com/chrisfilo/urbansound8k)


## Repository structure

```
├── notebooks          <- experiments on model
├── results            <- classification results based on 3D signal
├── script             <- preprocessing scripts
├── tiny_data          <- sample dataset
└── README.md
```

## Data Description

This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy. For a detailed description of the dataset and how it was compiled please refer to our paper.
All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results reported in the article above.

In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.




## Pipeline


In this project, we perform classification on both 1D and 3D audio signal. So we also try different preprocessing methods and algorithms respectively. 

### Preprocessing

For 1d signal, we simply load the audio file into digital format, and take non-overlapping sliding window to get fixed size audio clips. 

But to get the 3D signal, first we load the audio with double channel instead of single channel, and we have an extra step in between is to transform audio signal into mel-scaled spectrogram, which has been used widely in audio analysis. Then we take overlapping sliding window on it to get the data. 

The main structure of preprocessing are similiar, for example we filter audio with less than 1 second, and only consider audio with 44100 Hz. This means the data size reduced a lot from the original one.



### Model 

In the 1D signal model we used 4 convolution layers with batch normalization and max pooling after each convolution. The first convolution layer has a kernel size of 240, which allows the model to create filters that specifically match key parts of the audio signal. We also used audio augmentations like noise injection and pitch shift.

For the 3D signal CNN models, we created 3 different models. One is the with the raw sliced data without augmentations, one with augmentation such as fade in and out, and frequency masking, and last one is with double speed. We had to change the dimension slightly for the model with double speed since time is shorten. 


### Results

We were only able to achieve 45% validation accuracy for the 1D model, but for 3D we reached 77% validation accuracy. This might be because the 3D model mightve been less prone to overfitting since it was trained on the mel spectrogram data, which contains less noise and more important information than the raw audio.


## Future Work

1. Use pretrained resnet for 3D pipeline
2. Try processing data without slicing
3. Tune 1D CNN hyperparameters
4. Use confusion matrix to gain insight about predictions




