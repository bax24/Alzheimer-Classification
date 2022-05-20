# Alzheimer-Classification

model.py : contains all the models tried out, for now NetworkA uses 3D CNN and NetworkB is using Vision Transformer from MONAI (ViT)

TrainModelCNN.py : contains the train and test functions

main.py : contains the functions able to load the data, the setting of hyperparameters and the plot function for train and test loss

Normalizing the data was done in the Skull_stripped notebook by using the method implemented by https://github.com/jcreinhold/intensity-normalization


```
ssh lbackes@master.alan.priv
```

```
bax@MacBook-Pro-de-Lucas-3 Figures % scp -r  lbackes@master.alan.priv:/home/lbackes/mri_data/Figures ./
```

```
bax@MacBook-Pro-de-Lucas-3 Thesis % scp -r main.py lbackes@master.alan.priv:/home/lbackes/mri_data/
```


