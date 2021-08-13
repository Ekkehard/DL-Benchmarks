# Python Implementation: MPI weather forecasting benchmark
# -*- coding: utf-8 -*-
##
# @file       mpiWeather.py
#
# @version    1.0.0
#
# @par Purpose
#             Run a MPI Jena weather classification task using keras.
#
# @par Comments
#             This is another experiment from Chollet's book featuring a
#             temperature prediction task using a Gated Recurrent Unit (GRU)
#             input layer.
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
#   Wed Jul 17 2019 | Ekkehard Blanz | converted from Chollet's book
#                   |                |

import os
import time
import numpy as np

from keras import models
from keras import layers

def prepData( originalDatasetDir, trainSize ):

    fname = os.path.join( originalDatasetDir, 'mpi_roof_2009_2016.csv' )

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    if lines[-1]:
        lines = lines[1:]
    else:
        lines = lines[1:-1]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    mean = float_data[:trainSize].mean(axis=0)
    float_data -= mean
    std = float_data[:trainSize].std(axis=0)
    float_data /= std

    return float_data


def generator( data, lookback, delay, min_index, max_index,
               shuffle=False, batch_size=128, step=6 ):
    """!
    @param data The original array of floating-point data, which you
                normalized above
    @param lookback How many timesteps back the input data should go
    @param delay How many timesteps in the future the target should be
    @param min_index and max_index Indices in the data array that delimit which
           timesteps to draw from. This is useful for keeping a segment of the
           data for validation and another for testing
    @param shuffle Whether to shuffle the samples or draw them in chronological
           order
    @param batch_size The number of samples per batch
    @param step—The period, in timesteps, at which you sample data. You’ll set
           it to 6 in order to draw one data point every hour
    """

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets



def testRun( dtype ):

    lookback = 1440  # ten days
    step = 6         # one hour
    delay = 144      # one day - which element to predict
    batch_size = 128
    epochs = 10
    trainSize = 200000
    validationSize = 100000

    float_data = prepData( "../../Data/mpiJenaClimate", trainSize )
    testSize = len( float_data ) - (trainSize + validationSize + delay) - 1

    # customizations
    #trainSize = 0
    validationSize = 0
    #testSize = lookback + batch_size + 1
    #testSize = 0

    last = 0
    if trainSize:
        first = 0
        last = trainSize
        train_gen = generator( float_data,
                               lookback=lookback,
                               delay=delay,
                               min_index=first,
                               max_index=last,
                               shuffle=True,
                               step=step,
                               batch_size=batch_size )

    if validationSize:
        first = last
        last = first + validationSize
        val_gen = generator( float_data,
                             lookback=lookback,
                             delay=delay,
                             min_index=first,
                             max_index=last,
                             step=step,
                             batch_size=batch_size)

    if testSize:
        first = last
        last = first + testSize
        test_gen = generator( float_data,
                              lookback=lookback,
                              delay=delay,
                              min_index=first,
                              max_index=last,
                              step=step,
                              batch_size=batch_size )




    if validationSize:
        val_steps = validationSize- lookback
    else:
        val_steps = 0

    if testSize:
        test_steps = testSize - lookback
    else:
        test_steps = 0

    network = models.Sequential()
    network.add( layers.GRU( 32, input_shape=(None, float_data.shape[-1]) ) )
    network.add( layers.Dense( 1 ) )


    network.compile( optimizer="rmsprop", loss="mae" )

    if trainSize:
        start = time.time()
        if val_steps:
            history = network.fit_generator( train_gen,
                                             steps_per_epoch=500,
                                             epochs=epochs,
                                             validation_data=val_gen,
                                             validation_steps=val_steps )
            # do whatever analysis with history
        else:
            network.fit_generator( train_gen,
                                   steps_per_epoch=500,
                                   epochs=epochs )
        trainingTime = time.time() - start
    else:
        trainingTime = 0

    if testSize:
        start = time.time()
        testLoss = network.evaluate_generator( test_gen,
                                               steps=(test_steps // batch_size),
                                               verbose=1 )
        testAccuracy = None # not a classification task
        testTime = time.time() - start
    else:
        testTime = None
        testAccuracy = 0

    return (trainSize, testSize,
            trainingTime, testTime, testAccuracy, network)
