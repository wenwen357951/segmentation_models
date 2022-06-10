import os
import cv2
import keras
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm

# Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
segmentation_models could also use `tf.keras` if you do not have Keras installed 
or you could switch to other framework using `sm.set_framework('tf.keras')`
"""
sm.set_framework('tf.keras')

# Global Variable
DATA_DIR = '/app/examples/data/ICH'
OUTPUT_DIR = '/app/examples/output/normal-psp'
os.makedirs(OUTPUT_DIR, exist_ok=True)
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['ich']
LR = 0.0001
EPOCHS = 50


# Download CamVid Dataset
def download_dataset():
    # load repo with data if it is not exists
    if not os.path.exists(DATA_DIR):
        print('Loading data...')
        os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
        print('Done!')


# helper function for data visualization
def visualize(et="", fn="tmp.png", **images):
    """PLot images in one row."""
    num_of_images = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, num_of_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        if i == 2:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}-{i}-{et}-{fn}"))
            plt.close()
    # plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def display_form_dataset(img_dir: str, annot_dir: str, classes: [str], name, img_index=0, augmentation=None):
    # Let's look at data we have
    _dataset = Dataset(
        img_dir,
        annot_dir,
        classes=classes,
        augmentation=augmentation
    )

    # get some sample
    _img, _mask = _dataset[img_index]
    visualize(
        fn=name,
        image=_img,
        cars_mask=_mask[..., 0].squeeze(),
        sky_mask=_mask[..., 1].squeeze(),
        background_mask=_mask[..., 2].squeeze(),
    )


def get_training_parameters():
    # # case for binary and multiclass segmentation
    # Multiclass
    n_classes = len(CLASSES) + 1
    if len(CLASSES) == 1:
        # Binary
        n_classes = 1

    # Activation Type
    activation = 'softmax'
    if n_classes == 1:
        activation = "sigmoid"

    # Create Model
    model = sm.PSPNet(
        BACKBONE,
        classes=n_classes,
        activation=activation
    )

    return model, n_classes, activation


def train():
    """ Train Dataset """
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    """ Valid Dataset """
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    """ Test Dataset """
    # display_form_dataset(x_train_dir,
    #                      y_train_dir,
    #                      img_index=5,
    #                      classes=['car', 'pedestrian']
    #                      )

    """ Test Augmentation Dataset """
    # Lets look at augmented data we have
    # display_form_dataset(x_train_dir,
    #                      y_train_dir,
    #                      img_index=12,
    #                      classes=["car", "sky"],
    #                      augmentation=get_training_augmentation()
    #                      )

    """ Get Backbone preprocessing """
    preprocess_input = sm.get_preprocessing(BACKBONE)

    """ define network parameters """
    model, n_classes, _ = get_training_parameters()

    # define optimizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    if n_classes == 1:
        sm.losses.BinaryFocalLoss()

    total_loss = dice_loss + (1 * focal_loss)

    # actually total_loss can be imported directly from library,
    # above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        # augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, "training.png"))


def inference():
    """ Val Dataset """
    x_val_dir = os.path.join(DATA_DIR, 'val')
    y_val_dir = os.path.join(DATA_DIR, 'valannot')

    """ Test Dataset """
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    """ Get Backbone preprocessing """
    preprocess_input = sm.get_preprocessing(BACKBONE)

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    val_dataset = Dataset(
        x_val_dir,
        y_val_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    datasets = [val_dataset, test_dataset]

    val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
    dataloaders = [val_dataloader, test_dataloader]

    """ define network parameters """
    model, n_classes, _ = get_training_parameters()

    # define optimizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    if n_classes == 1:
        sm.losses.BinaryFocalLoss()

    total_loss = dice_loss + (1 * focal_loss)

    # actually total_loss can be imported directly from library,
    # above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # load best weights
    model.load_weights('./best_model.h5')

    evaluate_type = ["val", "test"]
    for idx, dataloader in enumerate(dataloaders):
        scores = model.evaluate(dataloader)

        print(f"Start process {evaluate_type[idx]}")

        # actually total_loss can be imported directly from library,
        # above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        print("Loss: {:.5}".format(scores[0]))
        for metric, value in zip(metrics, scores[1:]):
            print("mean {}: {:.5}".format(metric.__name__, value))

        # n = 5
        # ids = np.random.choice(np.arange(len(test_dataset)), size=n)

        for i in np.arange(len(datasets[idx])):
            image, gt_mask = datasets[idx][i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image).round()

            print(f"process {evaluate_type[idx]} Image - ({i}/{len(datasets[idx])})")

            visualize(
                et=evaluate_type[idx],
                fn=datasets[idx].get_basename(i),
                image=denormalize(image.squeeze()),
                gt_mask=gt_mask[..., 0].squeeze(),
                pr_mask=pr_mask[..., 0].squeeze(),
            )


def main():
    # Download CamVid Test Dataset
    # download_dataset()
    train()
    inference()


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']
    CLASSES = ['ich']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_basename(self, i):
        return os.path.basename(self.images_fps[i])

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=512, width=512, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(512, 512)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


if __name__ == '__main__':
    main()
