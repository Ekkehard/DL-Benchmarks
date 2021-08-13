# Python Implementation: Dogs vs Cats image recognition benchmark
# -*- coding: utf-8 -*-
##
# @file       dogsVsCats.py
#
# @version    1.0.0
#
# @par Purpose
#             Run the Kaggle dogs vs cats experiment using keras.
#
# @par Comments
#             This experiment is also from Chollet's book using 150 by 150 pixel
#             color JPEG images with a net using convolution layers.
#
#             This is Python 3 code!

# Known Bugs: none
#
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2019
#
# @copyright  See COPYING file that comes with this distribution
#
# File history:
#
#      Date         | Author         | Modification
#  -----------------+----------------+------------------------------------------
#   Sat Jul 13 2019 | Ekkehard Blanz | converted from Chollet's book
#                   |                |

import os
import time
import shutil
import uuid

from keras import models
from keras import layers
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

def prepData( originalDatasetDir, size ):
    """!
    @brief Prepare the dogs-versus-cats dataset.

    The directory structure that the ImageDataGenerator expects is a directory
    for each category - this is what this function provides.  Other than the
    one in Chollet's book, this one creates only symlinks rather than copying
    half a GB of data around and it creates the new structure in /tmp where
    everybody has write access.  Lastly, the creation of a validation data set
    and the test data set are optional and controlled via the tuple parameter
    size.  At any rate, each set will have exactly as many images from cats as
    it has from dogs.  If one set cardinality is zero, the corresponding
    directory name is empty and the directory will not be created.
    @param originalDatasetDir root directory where the Kaggle dataset resides
    @param size tuple with sizes of test, validation and test set
    @return (base_dir, train_dir, validation_dir, test_dir) tuple
    """

    if not os.path.isdir( originalDatasetDir ) or \
       not os.path.isdir( os.path.join( originalDatasetDir, "train" ) ):
        raise ValueError( "Error: Wrong original dataset direcotry specified"
                          "{0}".format( originalDatasetDir ) )

    original_dataset_dir = os.path.join( os.path.abspath( originalDatasetDir ),
                                         "train" )

    tmpdir = os.getenv( "TEMP", "/tmp" )

    base_dir = os.path.join( tmpdir, str( uuid.uuid4() ) )

    if size[0] + size[1] + size[2] > len( os.listdir( original_dataset_dir ) ):
        raise ValueError( "Error: Not enough data for size {0}".format( size ) )

    trainSize = size[0] // 2
    validationSize = size[1] // 2
    testSize = size[2] // 2

    os.mkdir(base_dir)

    if trainSize:
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)

        train_cats_dir = os.path.join(train_dir, 'cats')
        os.mkdir(train_cats_dir)

        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.mkdir(train_dogs_dir)
    else:
        train_dir = ""

    if validationSize:
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)

        validation_cats_dir = os.path.join(validation_dir, 'cats')
        os.mkdir(validation_cats_dir)

        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.mkdir(validation_dogs_dir)
    else:
        validation_dir = ""

    if testSize:
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)

        test_cats_dir = os.path.join(test_dir, 'cats')
        os.mkdir(test_cats_dir)

        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.mkdir(test_dogs_dir)
    else:
        test_dir = ""

    last = 0
    if trainSize:
        first = last
        last = first + trainSize
        fnames = ['cat.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            os.symlink(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            os.symlink(src, dst)

    if validationSize:
        first = last
        last = first + validationSize
        fnames = ['cat.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            os.symlink(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            os.symlink(src, dst)

    if testSize:
        first = last
        last = first + testSize
        fnames = ['cat.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            os.symlink(src, dst)

        fnames = ['dog.{}.jpg'.format(i) for i in range(first, last)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            os.symlink(src, dst)

    return (base_dir, train_dir, validation_dir, test_dir)


def testRun( dtype ):

    trainSize = 2000
    testSize = 1000
    batchSize = 20

    baseDir, trainDir, validationDir, testDir = prepData(
        "../../Data/dogs-vs-cats", (trainSize, 0, testSize) )

    datagen = ImageDataGenerator( rescale=1/255, dtype=dtype )

    trainGenerator = datagen.flow_from_directory(
        trainDir,
        target_size=(150, 150),
        batch_size=batchSize,
        class_mode="binary" )

    testGenerator = datagen.flow_from_directory(
        testDir,
        target_size=(150, 150),
        batch_size=batchSize,
        class_mode="binary" )

    network = models.Sequential()

    # Note that conv2D layers require a different input_shape parameter than
    # Dense layers (i.e. (rows, cols, channels) versus regular np shape tuples)

    # The first layers computes 32 features in 3 x 3 windows non-strided
    network.add( layers.Conv2D( 32, (3, 3), activation="relu",
                                input_shape=(150, 150, 3) ) )
    # non-linear (find max) reduction by factor of 2 in both directions with
    # stride (2, 2) (the default stride is same size as the reduction)
    network.add( layers.MaxPooling2D( (2, 2) ) )

    # out of these 32 features, this layer computes 64 features in 3 x 3 windows
    network.add( layers.Conv2D( 64, (3, 3), activation="relu" ) )
    network.add( layers.MaxPooling2D( (2, 2) ) )
    network.add( layers.Conv2D( 128, (3, 3), activation="relu" ) )
    network.add( layers.MaxPooling2D( (2, 2) ) )
    network.add( layers.Conv2D( 128, (3, 3), activation="relu" ) )
    network.add( layers.MaxPooling2D( (2, 2) ) )

    network.add( layers.Flatten() )
    network.add( layers.Dense( 512, activation="relu" ) )
    network.add( layers.Dense( 1, activation="sigmoid" ) )


    network.compile( optimizer=optimizers.RMSprop( lr=1.e-4 ),
                     loss="binary_crossentropy",
                     metrics=["accuracy"] )

    start = time.time()
    # 100 steps times batch size of 20 yields all 2000 training samples
    network.fit_generator( trainGenerator,
                           steps_per_epoch=(trainSize // batchSize),
                           epochs=15 )
    trainingTime = time.time() - start

    start = time.time()
    # 50 steps times batch size of 20 yields all 1000 training samples
    testLoss, testAccuracy = \
        network.evaluate_generator( testGenerator,
                                    steps=(testSize // batchSize) )
    testTime = time.time() - start

    shutil.rmtree( baseDir )

    return (trainSize, testSize, trainingTime, testTime, testAccuracy, network)

