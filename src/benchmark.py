#!/usr/bin/env python3

# Python Implementation: benchmark arbitrary test runs.
# -*- coding: utf-8 -*-
##
# @file       benchmark.py
#
# @version    1.3.1
#
# @par Purpose
#             Run a Python script using keras and tensorflow as a benchmark and
#             create a log file with all the relevant benchmark information.
#
# @par Synopsis:
#                 benchmark.py <module> [<exp. number>] [<mlc device]
#             where exp. number is the number of the experiment and mlc device
#             is one of "cpu" "gpu" or "any" where "any" lets TensorFlow decide
#             which computational device to use, which is the default.  This is
#             only used on the Mac.  The sequence of the optional parameters is
#             arbitrary.  If the experiment number is not given, it is not used
#             as part of the log file name.  If the architecture has no GPU, the
#             mlc device parameter is ignored if given.
#
# @par Comments
#             The Python functions that this benchmark wrapper runs are supposed
#             to be self-contained and only use the datatype for the
#             measurements as an input and to produce a tuple consisting of
#             training size, test size, training time, test time, test accuracy,
#             and the trained network model as an output.  The Python scripts
#             are expected to reside in the same directory as this script.  The
#             logs will be placed in a parallel logs sub-directory, which is
#             expected to exist and be writeable. Since data that the scripts
#             are using could be on a mounted disk, the functions should avoid
#             writing to the data directory but should rather use $TEMP (or
#             /tmp) to write temporary data, should that be necessary.
#
#             This is Python 3 code!
#
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
#   Sat Jul 06 2019 | Ekkehard Blanz | extracted from individual benchmarks
#   Sun Jul 21 2019 | Ekkehard Blanz | changed directory structure of log files
#   Sun Jul 28 2019 | Ekkehard Blanz | added addOn parameter
#   Wed Jun 30 2021 | Ekkehard Blanz | added Mac M1 support and call to
#                   |                | mlcompute.set_mlc_device()
#   Thu Aug 19 2021 | Ekkehard Blanz | caught exception from missing mlcompute
#                   |                |

import sys
import os

import cpuinfo
import psutil
import tensorflow as tf
try:
    from tensorflow.python.compiler.mlcompute import mlcompute
    haveMlcompute = True
except ModuleNotFoundError:
    haveMlcompute = False


log = ""


def addSummary( line ):
    global log
    log += line + "\n"
    return


# take care of the idiosyncrasies of the different architectures
info = cpuinfo.get_cpu_info()
try:
    vendor = info["vendor_id"]
    arch = info["arch"]
    brand = info["brand"]
    freqAdvertised = info["hz_advertised"]
    if arch == "ARM_8":
        vendor = "NVIDIA"
        hasGPU = True
        dtype = "float16"
    else:
        hasGPU = False
        dtype = "float32"
except KeyError:
    # Apple and Raspberry Pi don't have vendor_id key
    try:
        vendor = info["brand_raw"][0:5]
        arch = info["brand_raw"][6:]
        brand = info["brand_raw"]
        freqAdvertised = "??? Hz" # not available on M1
        hasGPU = True
        dtype = "float32"
    except KeyError:
        # this may be a stretch - but it works in my setting where RPi is the
        # only one that has neither vendor_id nor the brand_raw key set
        vendor = "RaspberryPi"
        arch = info["arch"]
        brand = info["brand"]
        freqAdvertised = info["hz_advertised"]
        hasGPU = False
        dtype = "float32"

# parse command line arguments

if len( sys.argv ) < 2:
    print( "benchmark requires the script to be benchmarked as argument" )
    sys.exit( 1 )
moduleName = sys.argv[1]
if moduleName.endswith( ".py" ):
    moduleName = moduleName[:-3]

addOn = ""
deviceName = "any"
if len( sys.argv ) > 2:
    try:
        addOn = "_" + str( int( sys.argv[2] ) )
    except ValueError:
        if hasGPU:
            deviceName = sys.argv[2].lower()

if len( sys.argv ) > 3:
    try:
        addOn = "_" + str( int( sys.argv[3] ) )
    except ValueError:
        if hasGPU:
            deviceName = sys.argv[3].lower()

if hasGPU:
    if deviceName in ["gpu", "cpu", "any"]:
        if haveMlcompute:
            mlcompute.set_mlc_device( device_name=deviceName )
            addOn = "_" + deviceName + addOn
    else:
        print( "ERROR: Wrong command line argument: ", sys.argv[3] )
        sys.exit( 1 )

exec( "from " + moduleName + " import testRun" )

trainingSize, testSize, trainingTime, testTime, testAccuracy, network = \
    testRun( dtype )

log += "Running " + moduleName + " on " + brand + ", "
log += "{0} bits\n".format( info["bits"] )
log += "with {0} cores, ".format( os.cpu_count() )
log += "running at " + freqAdvertised + "\n"
log += "Installed memory: " \
       "{0} GB\n".format( round( psutil.virtual_memory().total / 1024**3 ) )
log += "Floatingpoint precision: " + dtype + "\n"
log += "GPU acceleration is "
if hasGPU:
    log += "available "
    if deviceName == "gpu":
        log += "and"
    elif deviceName == "cpu":
        log += "but not"
    elif deviceName == "any":
        log += "but may not get"
    log += " used"
else:
    log += "not available"
log += "\nUsing TensorFlow Version "
log += tf.__version__
log += "\n\n\n"

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

filename = "../logs/" + moduleName + "." + vendor + arch + \
           addOn + ".log"

print( "Writing to filename: ", filename )
f = open( filename, "w" )
f.write( log )
f.close()

sys.exit( 0 )
