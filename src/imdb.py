# Python Implementation: IMDB benchmark
# -*- coding: utf-8 -*-
##
# @file       imdb.py
#
# @version    1.0.1
#
# @par Purpose
#             Run a IMDB movie review classification task using keras.
#
# @par Comments
#             This is the second experiment from Chollet's book featuring a
#             binary text classification task (positive or negative movie
#             reviews).
#
#             This is Python 3 code!

# Known Bugs: none
#
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2019-2021
#
# @copyright  See COPYING file that comes with this distribution
#
# File history:
#
#      Date         | Author         | Modification
#  -----------------+----------------+------------------------------------------
#   Sat Jul 06 2019 | Ekkehard Blanz | converted from Chollet's book
#   Thu Jul 01 2021 | Ekkehard Blanz | omitted pickle-fix on Mac
#                   |                |

from sys import platform

import time
import numpy as np

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

from keras.datasets import imdb


def vectorizeSequences( sequences, dtype, dimension=10000 ):
    results = np.zeros( (len( sequences ), dimension), dtype=dtype )
    for i, sequence in enumerate( sequences ):
        results[i, sequence] = 1.
    return results


def testRun( dtype ):

    if platform != "darwin":
        # save np.load on everything but Mac, which takes care of that in their
        # own TensorFlow
        npLoadOld = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: npLoadOld( *a, allow_pickle=True, **k )


    # call load_data with allow_pickle implicitly set to true
    (trainData, trainLabels), (testData, testLabels) = \
        imdb.load_data( num_words=10000 )

    if platform != "darwin":
        # restore np.load for future normal usage
        np.load = npLoadOld

    xTrain = vectorizeSequences( trainData, dtype )
    xTest = vectorizeSequences( testData, dtype )

    yTrain = np.asarray( trainLabels ).astype( dtype )
    yTest = np.asarray( testLabels ).astype( dtype )

    network = models.Sequential()

    network.add( layers.Dense( 16, activation="relu", input_shape=(10000,) ) )
    network.add( layers.Dense( 16, activation="relu" ) )
    network.add( layers.Dense( 1, activation="sigmoid" ) )

    #xVal = xTrain[:10000]
    #partialXtrain = xTrain[10000:]
    #yVal = yTrain[:10000]
    #partialYtrain = yTrain[10000:]


    network.compile( optimizer=optimizers.RMSprop( lr=0.001 ),
                     loss=losses.binary_crossentropy,
                     metrics=[metrics.binary_accuracy] )

    start = time.time()
    network.fit( xTrain, yTrain, epochs=4, batch_size=512 )
    trainingTime = time.time() - start

    start = time.time()
    # loss is e.g. least squares error, accuracy is after non-linear decision
    testLoss, testAccuracy = network.evaluate( xTest, yTest )
    testTime = time.time() - start

    return (len( xTrain ), len( xTest ),
            trainingTime, testTime, testAccuracy, network)
