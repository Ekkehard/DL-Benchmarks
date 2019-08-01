# Python Implementation: mnist benchmark - one-dimensional approach
# -*- coding: utf-8 -*-
##
# @file       mnist1D.py
#
# @version    1.0.1
#
# @par Purpose
#             Run a MNIST handwritten digits classification task using keras.
#
# @par Comments
#             This is the first experiment from Chollet's book flattening the
#             images of the digits into vectors.
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
#   Tue May 21 2019 | Ekkehard Blanz | created
#   Sat Jul 06 2019 | Ekkehard Blanz | converted to benchmarkable function
#                   |                |

import time

from keras import models
from keras import layers
from keras.utils import to_categorical

from keras.datasets import mnist

def testRun( dtype ):

    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    # convert 28 x 28 image matrices into 784 x 1 vectors
    trainImages = trainImages.reshape( (60000, 28*28) )
    trainImages = trainImages.astype( dtype ) / 255

    testImages = testImages.reshape( (10000, 28*28) )
    testImages = testImages.astype( dtype ) / 255

    trainLabels = to_categorical( trainLabels ).astype( dtype )
    testLabels = to_categorical( testLabels ).astype( dtype )

    network = models.Sequential()

    # "densely connected" layers is keras parlor for "fully connected" layers
    network.add( layers.Dense( 512, activation="relu", input_shape=(28*28,) ) )
    # the softmax activatio function gives us probabilities that sum up to 1
    network.add( layers.Dense( 10, activation="softmax" ) )


    network.compile( optimizer="rmsprop", loss="categorical_crossentropy",
                     metrics=["accuracy"] )

    start = time.time()
    network.fit( trainImages, trainLabels, epochs=5, batch_size=128 )
    trainingTime = time.time() - start

    start = time.time()
    # loss is e.g. least squares error, accuracy is after non-linear decision
    testLoss, testAccuracy = network.evaluate( testImages, testLabels )
    testTime = time.time() - start

    return (len( trainImages ), len( testImages ),
            trainingTime, testTime, testAccuracy, network)
