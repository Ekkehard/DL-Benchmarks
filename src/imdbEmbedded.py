# Python Implementation: IMDB with embedded word vectors benchmark
# -*- coding: utf-8 -*-
##
# @file       imdbEmbedded.py
#
# @version    1.0.1
#
# @par Purpose
#             Run a IMDB movie review classification task with embedded word
#             vectors using keras.
#
# @par Comments
#             This is another experiment from Chollet's book featuring a
#             binary text classification task (positive or negative movie
#             reviews) but this time using embedded word vectors.
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
from keras import preprocessing

from keras.datasets import imdb


def testRun( dtype ):

    # size of vocabulary
    maxFeatures = 10000
    # embedding dimension
    dim = 8
    # use only at most maxLen words from each review
    maxLen = 50

    if platform != "darwin":
        # save np.load on everything but Mac, which takes care of that in their
        # own TensorFlow
        npLoadOld = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: npLoadOld( *a, allow_pickle=True, **k )

    # call load_data with allow_pickle implicitly set to true
    (trainData, trainLabels), (testData, testLabels) = \
        imdb.load_data( num_words=maxFeatures )

    if platform != "darwin":
        # restore np.load for future normal usage
        np.load = npLoadOld

    xTrain = preprocessing.sequence.pad_sequences( trainData,
                                                   dtype=dtype,
                                                   maxlen=maxLen )
    xTest = preprocessing.sequence.pad_sequences( testData,
                                                   dtype=dtype,
                                                   maxlen=maxLen )

    yTrain = np.asarray( trainLabels ).astype( dtype )
    yTest = np.asarray( testLabels ).astype( dtype )

    network = models.Sequential()
    network.add( layers.Embedding( maxFeatures, dim, input_length=maxLen ) )

    network.add( layers.Flatten() )
    network.add( layers.Dense( 1, activation="sigmoid" ) )


    network.compile( optimizer="rmsprop",
                     loss="binary_crossentropy",
                     metrics=["acc"] )

    start = time.time()
    network.fit( xTrain, yTrain, epochs=10, batch_size=32 )
    trainingTime = time.time() - start

    start = time.time()
    # loss is e.g. least squares error, accuracy is after non-linear decision
    testLoss, testAccuracy = network.evaluate( xTest, yTest )
    testTime = time.time() - start

    return (len( xTrain ), len( xTest ),
            trainingTime, testTime, testAccuracy, network)
