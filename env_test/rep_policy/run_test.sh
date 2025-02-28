#!/bin/bash

# run unit test
../../src/cache_simulator.py -pdc $1/config -t $1/testAC.txt -f $1/result.txt

if diff -u "$1/result.txt" "$1/expected_result.txt" ; then
    echo "$1 test passed! "
else
    echo "$1 test failed!" 
    : 
fi