# Computer Vision Project: Gender Prediction

Vito Lin and John Lee

## Goal
We have two goals for our project. The first is to be able to take an image of a person and discern the perceived biological sex of the individual. We want to view intermediate feature maps in our model to potentially identify features that help classify biological sex. Hopefully, they help us discern which traits may be more masculine or feminine. Our second goal is to quantify model error that occurs when the model is trained on a certain ethnicity, but a different one is fed through. Paired with the intermediate feature maps, we could potentially identify which characteristics display/classify differently between ethnicities.
 

### Tasks:
In order to accomplish this goal we have to first gather a large dataset of faces with tags on the perceived gender of all the faces. We would want the data set to be consistent in terms of angle of face, illumination, and area that is cropped (or other image augmentations). From here we would want to feed into some machine learning algorithm these tagged photos and be able to discern the masculine and feminine features to predict the perceived gender of the face. 
    We also want to minimize the amount of features that may be associated with a feminine or masculine face that may not be from the actual face but perhaps artifacts from the sources we gather the faces from. This would just mean screening the pictures ahead of time to see if there are artifacts that exist (maybe something as silly as the word male or female written on the photos).

### Learning:
* Learn how to use machine learning in order to categorize images.
* Learn how to use machine learning for facial recognition.
* Learn how ethnicities might affect facial recognition and introduce model error.
* Learn how to curate good datasets for training, validation, and testing. 
* Learn about different machine learning algorithms in order to find an algorithm that creates the best results for our project.

## Datasets:
* Large data set or sets of faces with tagged perceived genders.
* Be able to preprocess these data sets to that way the are consistent to be used in our algorithm
* We want to gather a large amount of faces from different sites that hold faces and then preprocess them all to work within our model. We can then split the model into training, validation, and testing.


### Source
We will utilize the CelebA Dataset from Kaggle found here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download.
Furthermore, we also utilize the Fair Face dataset found here: https://github.com/joojs/fairface.

### Description
The original paper the CelebA dataset was created for is:
S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015

The CelebA dataset contains 202599 facial images of celebrities, cropped and aligned. Each image comes with 40 (-1, 1) binary attribute annotations (including Gender) and 5 landmark locations (eyes, nose, left/right mouth). Of the 202599 images, there are 10,177 unique anonymous identities.
The images are taken in many different backgrounds, with large variations in lighting, and are taken at no consistent angle. Each picture has a relatively unobstructed view of the celebrity's face and are taken with no consistent camera (focal lense, quality, color, etc)

The Fair Face dataset was created for use in :
Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).

The Fair Face dataset contains 108,501 images from 7 different race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino. Each image is cropped to a size of 150 x 150, with a padding of .25, focusing on an unobstructed view of an individual's face. Furthermore, each image is identified with either Male or Female binary identifiers and an age group. The images consist a balanced ethnicity dataset originating from YFCC-100M Flickr dataset.


### Implementation
CelebA:
Since some images contain distracting annotated features such as eyewear, we will likely reconsider their usage in the dataset.
Although the dataset provides a suggested train/val/test distribution roughly approximating to 80/10/10, if we remove certain images mentioned above, we will randomly repartition the remaining images into the same ratio. Furthermore, we may wish to treat the males and females as 2 different datasets. In that case, we will certianly repartition the data.

There will be little difference between the training and validation data sets as we are randomly sampling from the same source and dividing the total dataset to a 80/10/10 ratio (training, testing, validation). We want to keep the training and validation data sets consistent as to not bias the model in any way.

FairFace:
Since the data provides a relatively clear image of the individuals face, no real transformation/changes should be necessary. However, we need to implement a filter that allows us to train using only a certain ethnicity and test using a different ethnicity. 

### Preprocessing

See [PREPROCESSING](preprocessing/PREPROCESSING.md)

### Classification

See [CLASSIFICATION](classification/CLASSIFICATION.md)



### Installation
Clone github into local directory.
Navigate to root of git repository.
Run conda env create.
To utilize cuda, go to pytorch website [https://pytorch.org/] and install appropriate version of pytorch

Download Data and place in appropriate files:
    FairFace: https://github.com/joojs/fairface -> [padding = .25][fairface_label_train.csv][fairface_label_val.csv]
    CelebA: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download -> extract zip file

    Dir Structure
    data
        fairface-img-margin025-trainval
            train
                ...
            val
                ...
        img_align_celeba
            ...
        fairface_label_train.csv
        fairface_label_val.csv
        list_attr_celeba.csv
