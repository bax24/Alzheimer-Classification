# Alzheimer-Classification

model.py : contains all the models tried out, for now NetworkA uses 3D CNN and NetworkB is using Vision Transformer from MONAI (ViT)

TrainModelCNN.py : contains the train and test function 

main.py : contains the functions able to load the data, the setting of hyperparameters and the plot function for train and test loss

Last time I ran this code (so with the ViT model) I got those plots

```
ssh lbackes@master.alan.priv
```

```
bax@MacBook-Pro-de-Lucas-3 Figures % scp -r  lbackes@master.alan.priv:/home/lbackes/mri_data/Figures ./
```

```
bax@MacBook-Pro-de-Lucas-3 Thesis % scp -r main.py lbackes@master.alan.priv:/home/lbackes/mri_data/
```


At the moment the networks are trained and tested on a very very small dataset but the 3D CNN network still learns along epochs.
