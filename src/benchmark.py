#!/usr/bin/env python3
# Python Implementation: benchmark arbitrary test runs.
# -*- coding: utf-8 -*-
##
# @file       benchmark.py
#
# @version    1.2.0
#
# @par Purpose
#             Run a Python script using keras and tensorflow as a benchmark and
#             create a log file with all the relevant benchmark information.
#
# @par Comments
#             The Python functions that this benchmark wrapper runs are supposed
#             to be self-contained and only use the datatype for the
#             measurements as an input and to produce a tuple consisting of
#             training size, test size, training time, test time, test accuracy,
#             and the trained network model as an output.  The Python scripts
#             are expected to reside in the same directory as this script.  The
#             logs will be placed in a logs sub-directory, which is expected to
#             exist and be writeable. Since data the scripts are using are
#             likely on a mounted disk, the functions should avoid writing to
#             the data directory but should rather use $TEMP (or /tmp) to write
#             temporary data, should that be necessary.
#
#             This is Python 3 code!
#
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
#   Sat Jul 06 2019 | Ekkehard Blanz | extracted from individual benchmarks
#   Sun Jul 21 2019 | Ekkehard Blanz | changed directory structure of log files
#   Sun Jul 28 2019 | Ekkehard Blanz | added addOn parameter
#                   |                |

import sys
import os

import cpuinfo
import psutil
import tensorflow as tf

log = ""


def addSummary( line ):
    global log
    log += line + "\n"
    return


info = cpuinfo.get_cpu_info()

if info["arch"] == "ARM_8":
    # This is for the NVIDIA Jetson Nano - it uses half precision in its GPU
    dtype = "float16"
else:
    # everybody else uses single precision here - CPU or GPU
    dtype = "float32"

if len( sys.argv ) < 2:
    print( "benchmark requires the script to be benchmarked as argument" )
    sys.exit( 1 )

if len( sys.argv ) > 2:
    addOn = "_" + sys.argv[2]
else:
    addOn = ""

moduleName = sys.argv[1]
if moduleName.endswith( ".py" ):
    moduleName = moduleName[:-3]


exec( "from " + moduleName + " import testRun" )


trainingSize, testSize, trainingTime, testTime, testAccuracy, network = \
    testRun( dtype )


log += "Running " + moduleName + " on " + info["brand"] + ", "
log += "{0} bits\n".format( info["bits"] )
log += "with {0} cores, ".format( os.cpu_count() )
log += "running at " + info["hz_advertised"] + "\n"
log += "Installed memory: " \
       "{0} GB\n".format( round( psutil.virtual_memory().total / 1024**3 ) )
log += "Floatingpoint precision: " + dtype + "\n"
log += "NVIDIA GPU acceleration is "
if not tf.test.is_gpu_available():
    log += "not "
log += "available\n\n\n"
log += "Training size: {0:7d} samples\n".format( trainingSize )
log += "Test size:     {0:7d} samples\n".format( testSize )
log += "Training time: {0:7.3f} s\n".format( trainingTime )
log += "Test time:     {0:7.3f} s\n".format( testTime )
if testAccuracy is not None:
    log += "Classification accuracy on test data: " \
        "{0:4.2f} %\n".format( testAccuracy * 100 )
log += "\n\nNet architecture:\n"
log += "Input Shape:  {0}\n\n".format( network.input_shape )
network.summary( print_fn=addSummary )

print( "\n\n\n" )
print( log )
print( "\n" )

try:
    vendor = info["vendor_id"]
except KeyError:
    # this may be a stretch - but it works in my setting where RPi is the only
    # one that doesn't have the vendor_id set
    vendor = "RaspberryPi"

filename = "../logs/" + moduleName + "." + vendor + info["arch"] + \
           addOn + ".log"
f = open( filename, "w" )
f.write( log )
f.close()

sys.exit( 0 )
