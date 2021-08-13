# Python Implementation: mnist benchmark - two-dimensional approach
# -*- coding: utf-8 -*-
##
# @file       mnist2D.py
#
# @version    1.0.1
#
# @par Purpose
#             Run a MNIST handwritten digits classification task using keras.
#
# @par Comments
#             This is an improved MNIST hand printed character recognition
#             experiment and the fourth experiment from Chollet's book using the
#             full 2-D images of the digits and convolution layers.
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
#   Wed May 22 2019 | Ekkehard Blanz | converted from Chollet's book
#   Sat Jul 06 2019 | Ekkehard Blanz | converted to benchmarkable function
#                   |                |

import time

from keras import models
from keras import layers
from keras.utils import to_categorical

from keras.datasets import mnist

def testRun( dtype ):

    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    trainImages = trainImages.reshape( (60000, 28, 28, 1) )
    trainImages = trainImages.astype( dtype ) / 255

    testImages = testImages.reshape( (10000, 28, 28, 1) )
    testImages = testImages.astype( dtype ) / 255

    trainLabels = to_categorical( trainLabels ).astype( dtype )
    testLabels = to_categorical( testLabels ).astype( dtype )

    network = models.Sequential()

    # Note that conv2D layers require a different input_shape parameter than
    # Dense layers (i.e. (rows, cols, channels) versus regular np shape tuples)

    # The first layers computes 32 features in 3 x 3 windows non-strided
    network.add( layers.Conv2D( 32, (3, 3), activation="relu",
                                input_shape=( 28, 28, 1) ) )
    # non-linear (find max) reduction by factor of 2 in both directions with
    # stride (2, 2) (the default stride is same size as the reduction)
    network.add( layers.MaxPooling2D( (2, 2) ) )

    # out of these 32 features, this layer computes 64 features in 3 x 3 windows
    network.add( layers.Conv2D( 64, (3, 3), activation="relu" ) )
    network.add( layers.MaxPooling2D( (2, 2) ) )
    network.add( layers.Conv2D( 64, (3, 3), activation="relu" ) )
    # no more maxpooling neede - we are already down to 7 x 7 "images"

    network.add( layers.Flatten() )
    network.add( layers.Dense( 64, activation="relu" ) )
    network.add( layers.Dense( 10, activation="softmax" ) )


    network.compile( optimizer="rmsprop", loss="categorical_crossentropy",
                     metrics=["accuracy"] )

    start = time.time()
    network.fit( trainImages, trainLabels, epochs=5, batch_size=64 )
    trainingTime = time.time() - start

    start = time.time()
    testLoss, testAccuracy = network.evaluate( testImages, testLabels )
    testTime = time.time() - start

    return (len( trainImages ), len( testImages ),
            trainingTime, testTime, testAccuracy, network)

