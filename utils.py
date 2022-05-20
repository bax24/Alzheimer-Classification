import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nipype.interfaces import fsl
from dltk.io import preprocessing
from skimage import filters


def display(path, filename, cut, axial=False, coronal=False, sagitall=False):
    image = os.path.join(path, filename)
    img = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(img)

    if axial:
        plt.imshow(image[cut, :, :], cmap='gray')
    elif coronal:
        plt.imshow(image[:, cut, :], cmap='gray')
    elif sagitall:
        plt.imshow(image[:, :, cut], cmap='gray')

    plt.gca().invert_yaxis()
    plt.savefig('original.png')
    plt.show()

    print(image[:, :, cut].min())
    print(image[:, :, cut].max())
    print(image.shape)



def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    ''' This function resamples images to 1-mm isotropic voxels (default).

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


def register_and_save(filename, path, description, target, mm=1.0):
    """ Process the image name and copy the image to its
    corresponding destination folder.

    Parameters:
        filename -- Name of the image file (.nii)
        path -- The path were the image is located
        atlas -- Reference sitk image for registration
    """

    # separate the name of the file by '_'
    splitted_name = filename.strip().split('_')

    # sometimes residual MacOS files appear; ignore them
    if splitted_name[0] == '.DS': return

    # Save the image ID
    image_ID = splitted_name[-1][0:-4]
    #### IMPORTANT #############
    # the following three lines are used to extract the label of the image
    # ADNI data provides a description .csv file that can be indexed using the
    # image ID. If you are not working with ADNI data, then you must be able to
    # obtain the image label (AD/MCI/NC) in some other way
    # with the ID, index the information we need
    row_index = description.index[description['Image Data ID'] == image_ID].tolist()[0]
    # Get the row
    row = description.iloc[row_index]
    # Get the label
    label = row['Group']
    # print(label)
    # load in sitk format
    # prepare the origin path
    complete_file_path = os.path.join(path, filename)
    # load sitk image
    sitk_moving = sitk.ReadImage(complete_file_path)
    sitk_moving = resample_img(sitk_moving, out_spacing=[mm, mm, mm])
    array_moving = sitk.GetArrayFromImage(sitk_moving)
    # registrated = registrate(atlas, sitk_moving)
    # res_img = preprocessing.resize_image_with_crop_or_pad(array_moving, img_size=size, mode='symmetric')
    # res_img = sitk.GetImageFromArray(res_img)

    # prepare the destination path
    complete_new_path = os.path.join(target, label,
                                     filename)
    sitk.WriteImage(sitk_moving, complete_new_path)


def skull_strip_nii(original_img, destination_img, frac=0.5):
    ''' Practice skull stripping on the given image, and save
        the result to a new .nii image.
        Uses FSL-BET
        (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:)

        Parameters:
            original_img -- Original nii image
            destination_img -- The new skull-stripped image
            frac -- Fractional intensity threshold for BET
    '''

    btr = fsl.BET()
    btr.inputs.in_file = original_img
    btr.inputs.frac = frac
    btr.inputs.out_file = destination_img
    btr.cmdline
    resu = btr.run()


def trim(arr, mask):
    bounding_box = tuple(
        slice(np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))
    return arr[bounding_box]
