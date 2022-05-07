# Alzheimer-Classification

model.py : contains all the models tried out, for now NetworkA uses 3D CNN and NetworkB is using Vision Transformer from MONAI (ViT)

TrainModelCNN.py : contains the train and test function 

main.py : contains the functions able to load the data, the setting of hyperparameters and the plot function for train and test loss

Last time I ran this code (so with the ViT model) I got those plots

'''
bax@MacBook-Pro-de-Lucas-3 Figures % scp -r  lbackes@master.alan.priv:/home/lbackes/mri_data/Figures ./
'''

![training](https://user-images.githubusercontent.com/38333245/163365187-6a7847e8-683d-4e48-bff5-136348eb6d2c.png)


![testing](https://user-images.githubusercontent.com/38333245/163365168-e47ae726-a4e5-492e-91d8-d5f8189fcfc0.png)


At the moment the networks are trained and tested on a very very small dataset but the 3D CNN network still learns along epochs.
