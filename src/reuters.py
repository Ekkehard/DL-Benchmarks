# Python Implementation: Reuters benchmark
# -*- coding: utf-8 -*-
##
# @file       reuters.py
#
# @version    1.0.0
#
# @par Purpose
#             Run a Reuters newswires classification task using keras.
#
# @par Comments
#             This is the third experiment of Chollet's book featuring a text
#             classification task with 46 categories.
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
#   Sat Jul 06 2019 | Ekkehard Blanz | converted from Chollet's book
#                   |                |

import time
import numpy as np

from keras import models
from keras import layers
from keras.utils import to_categorical

from keras.datasets import reuters


def vectorizeSequences( sequences, dtype, dimension=10000 ):
    results = np.zeros( (len( sequences ), dimension), dtype=dtype )
    for i, sequence in enumerate( sequences ):
        results[i, sequence] = 1.
    return results


def testRun( dtype ):
    # save np.load
    npLoadOld = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: npLoadOld( *a, allow_pickle=True, **k )

    # call load_data with allow_pickle implicitly set to true
    (trainData, trainLabels), (testData, testLabels) = \
        reuters.load_data( num_words=10000 )

    # restore np.load for future normal usage
    np.load = npLoadOld

    xTrain = vectorizeSequences( trainData, dtype )
    xTest = vectorizeSequences( testData, dtype )

    trainLabels = to_categorical( trainLabels ).astype( dtype )
    testLabels = to_categorical( testLabels ).astype( dtype )

    network = models.Sequential()

    network.add( layers.Dense( 64, activation="relu", input_shape=(10000,) ) )
    network.add( layers.Dense( 64, activation="relu" ) )
    network.add( layers.Dense( 46, activation="softmax" ) )

    network.compile( optimizer="rmsprop",
                     loss="categorical_crossentropy",
                     metrics=["accuracy"] )

    start = time.time()
    network.fit( xTrain, trainLabels, epochs=9, batch_size=512 )
    trainingTime = time.time() - start

    start = time.time()
    # loss is e.g. least squares error, accuracy is after non-linear decision
    testLoss, testAccuracy = network.evaluate( xTest, testLabels )
    testTime = time.time() - start

    return (len( xTrain ), len( xTest ),
            trainingTime, testTime, testAccuracy, network)
