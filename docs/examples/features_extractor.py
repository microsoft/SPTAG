import glob
import numpy as np
from PIL import Image
from keras import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3


def get_filenames(glob_pattern, recursive=True):
    """Extracts list of filenames (full paths) based on specific glob path pattern.
    
    Parameters
    ----------
    glob_pattern : str
        Glob pattern for glob to extract filenames, eg. "directory/**/*.jpg"
    recursive : bool, optional
        Recursively search through subdirectories, by default True
    
    Returns
    -------
    list
        List of file paths
    """
    all_files = glob.glob(glob_pattern, recursive=recursive)
    print('Found %s files using pattern: %s' % (len(all_files), glob_pattern))
    return all_files


def expand2square(pil_img, background_color):
    """Function to pad an image to square using specific bg clr.
    
    Parameters
    ----------
    pil_img : PIL.Image
        Pillow Image object that should be processed
    background_color : int
        Integer value representing bg color
    
    Returns
    -------
    PIL.Image
        Square-padded image object
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_images(filenames, target_size=(200,200), color='RGB', bg_clr=0):
    """Reads image files from provided file paths list, applies square-padding,
    resizes all images into target size and returns them as a single numpy array
    
    Parameters
    ----------
    filenames : list
        List of image file paths
    target_size : tuple, optional
        Target size for all the images to be resized to, by default (200,200)
    color : str, optional
        Color mode strategy for PIL when loading images, by default 'RGB'
    bg_clr : int, optional
        Integer representing background color used for square-padding, by default 0
    
    Returns
    -------
    numpy.array
        Numpy array with resized images
    """
    imgs_list = []
    for filename in filenames:
        img = Image.open(filename).convert(color)
        im_square = expand2square(img, bg_clr)
        im_res = im_square.resize(target_size)
        imgs_list.append(np.array(im_res))

    return np.asarray(imgs_list)


def create_feat_extractor(base_model, pooling_method='avg'):
    """Creates a features extractor based on the provided base network.
    
    Parameters
    ----------
    base_model : keras.Model
        Base network for feature extraction
    pooling_method : str, optional
        Pooling method that will be used as the last layer, by default 'avg'
    
    Returns
    -------
    keras.Model
        Ready to use feature extractor
    """
    assert pooling_method in ['avg', 'max']
    
    x = base_model.output
    if pooling_method=='avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling_method=='max':
        x = GlobalMaxPooling2D()(x)
    model = Model(input=base_model.input, output=[x])

    return model


def extract_features(imgs_np, pretrained_model="resnet50", pooling_method='avg'):    
    """Takes in an array of fixed size images and returns features/embeddings
    returned by one of the selected pretrained networks.
    
    Parameters
    ----------
    imgs_np : numpy.array
        Numpy array of images
    pretrained_model : str, optional
        Name of the pretrained model to be used, by default "resnet50"
        ['resnet50', 'inception_v3', 'vgg19']
    pooling_method : str, optional
        Defines the last pooling layer that should be applied, by default 'avg'
        ['avg', 'max']
    
    Returns
    -------
    numpy.array
        Array of embeddings vectors. Each row represents embeddings for single input image
    """
    print('Input images shape: ', imgs_np.shape)
    pretrained_model = pretrained_model.lower()
    assert pretrained_model in ['resnet50', 'inception_v3', 'vgg19']
    assert pooling_method in ['avg', 'max']

    model_args={
        'weights': 'imagenet',
        'include_top': False,
        'input_shape': imgs_np[0].shape
        }

    if pretrained_model=="resnet50":
        base = ResNet50(**model_args)
        from keras.applications.resnet50 import preprocess_input
    elif pretrained_model=="inception_v3":
        base = InceptionV3(**model_args)
        from keras.applications.inception_v3 import preprocess_input
    elif pretrained_model=="vgg19":
        base = VGG19(**model_args)
        from keras.applications.vgg19 import preprocess_input

    feat_extractor = create_feat_extractor(base, pooling_method=pooling_method)

    imgs_np = preprocess_input(imgs_np)
    embeddings_np = feat_extractor.predict(imgs_np)
    print('Features shape: ', embeddings_np.shape)
    
    return embeddings_np


# if __name__ == "__main__":
#     filenames = get_filenames("101_ObjectCategories//**//*.*")
#     imgs_np = get_images(filenames, target_size=(200,200), color='RGB', bg_clr=0)
#     embeddings = extract_features(imgs_np, pretrained_model="resnet50")