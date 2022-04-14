import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import trainModel as t
import torch
from model import NetworkB
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from model import NetworkA
import trainModelCNN as f

def isAmstrong(num):
    som = []
    temp = num
    digit = 0
    while temp > 0:
        add = temp % 10
        som.append(add)
        temp //= 10
        digit += 1
    temp = sum(map(lambda x: pow(x, digit), som))
    if num == temp:
        print(num, 'is a Armstrong number')
    else:
        print(num, 'is not a Armstrong number')


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    ''' This function resamples images to 2-mm isotropic voxels (default).

        Parameters:
            itk_image -- Image in simpleitk format, not a numpy array
            out_spacing -- Space representation of each voxel

        Returns:
            Resulting image in simpleitk format, not a numpy array
    '''

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def registrate(sitk_fixed, sitk_moving, bspline=False):
    ''' Perform image registration using SimpleElastix.
        By default, uses affine transformation.

        Parameters:
            sitk_fixed -- Reference atlas (sitk .nii)
            sitk_moving -- Image to be registrated
                           (sitk .nii)
            bspline -- Whether or not to perform non-rigid
                       registration. Note: it usually deforms
                       the images and increases execution times
    '''

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk_fixed)
    elastixImageFilter.SetMovingImage(sitk_moving)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    if bspline:
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    return elastixImageFilter.GetResultImage()


def register_and_save(filename, path, atlas):
    ''' Process the image name and copy the image to its
        corresponding destination folder.

        Parameters:
            filename -- Name of the image file (.nii)
            path -- The path were the image is located
            atlas -- Reference sitk image for registration
    '''

    # separate the name of the file by '_'
    splitted_name = filename.strip().split('_')
    # sometimes residual MacOS files appear; ignore them
    if splitted_name[0] == '.': return

    # save the image ID
    image_ID = splitted_name[-1][1:-4]

    # sometimes empty files appear, just ignore them (macOS issue)
    if image_ID == '': return
    # transform the ID into a int64 numpy variable for indexing
    image_ID = np.int64(image_ID)

    #### IMPORTANT #############
    # the following three lines are used to extract the label of the image
    # ADNI data provides a description .csv file that can be indexed using the
    # image ID. If you are not working with ADNI data, then you must be able to
    # obtain the image label (AD/MCI/NC) in some other way
    # with the ID, index the information we need
    row_index = description.index[description['Image Data ID'] == image_ID].tolist()[0]
    # obtain the corresponding row in the dataframe
    row = description.iloc[row_index]
    # get the label
    label = row['Group']

    # prepare the origin path
    complete_file_path = os.path.join(path, filename)
    # load sitk image
    sitk_moving = sitk.ReadImage(complete_file_path)
    sitk_moving = resample_img(sitk_moving)
    registrated = registrate(atlas, sitk_moving)

    # prepare the destination path
    complete_new_path = os.path.join(REG_DB,
                                     label,
                                     filename)
    sitk.WriteImage(registrated, complete_new_path)


def check_palindrome(num):
    for i, val in enumerate(num):
        if val == num[len(num) - i - 1]:
            continue
        else:
            print('Not a palindrome')
            return False
    print('Palindrome')
    return True


def get_data(train_size, skull_stripped=False):
    source = 'Resampled'
    if skull_stripped:
        source = 'Skull_stripped'

    # Training dataset
    MRI_train = []
    labels_train = []

    # Testing dataset
    MRI_test = []
    labels_test = []

    diagnosis = ['AD', 'CN']
    for diag in diagnosis:
        index = 0
        for path, dirs, files in os.walk(os.path.join(source, diag)):
            # Divide both classes evenly between train and test dataset
            upper_bound_train = int(len(files) * train_size)
            for filename in files:
                image = os.path.join(source, diag, filename)
                sitk_image = sitk.ReadImage(image)

                # transform into a numpy array
                MRI = sitk.GetArrayFromImage(sitk_image)
                # add the Channel dimension
                MRI = MRI[np.newaxis, ...]
                # Put sitk in right dataset
                if index < upper_bound_train:
                    # Train section
                    MRI_train.append(MRI)
                    labels_train.append(diag)
                    index += 1
                else:
                    MRI_test.append(MRI)
                    labels_test.append(diag)

    labels_train = torch.tensor([1 if x == 'AD' else 0 for x in labels_train])
    labels_test = torch.tensor([1 if x == 'AD' else 0 for x in labels_test])
    train_db = list(zip(MRI_train, labels_train))
    test_db = list(zip(MRI_test, labels_test))

    return train_db, test_db


if __name__ == '__main__':

    # Parameters
    batch_size = 5
    learning_rate = 0.001
    epochs = 10

    # Define my model
    # ViT Backbone
    model = NetworkB(in_channel=1, out_channel=1, img_size=(96, 96, 48), pos_embed='conv')

    # 3D CNN
    #model = NetworkA(init_kernel=16)

    # Define my optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    # Define my Loss
    loss = nn.BCELoss() # Not used for ViT

    # Downloading datasets
    train_set, test_set = get_data(train_size=0.7, skull_stripped=False)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    train_loss = []
    test_loss = []

    for epoch in range(1, epochs):
        total_loss = f.train(model, train_loader, optimizer, epoch)
        train_loss.append(total_loss)
        print("Training ok !")
        total_loss = f.test(model, test_loader, epoch)
        test_loss.append(total_loss)

    # Plotting loss
    plt.Figure(figsize=(13, 5))
    plt.title('Training loss')
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.Figure(figsize=(13, 5))
    plt.title('Testing loss')
    plt.plot(test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


    # for nb, (mri, label) in enumerate(train_loader):
    #   print(label)
