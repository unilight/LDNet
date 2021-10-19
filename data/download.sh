#!/bin/bash

dataset=$1

if [ ${dataset} = "vcc2018" ]; then
    mkdir -p ${dataset}
    cd ${dataset}
    if [ ! -e ./.done ]; then
        wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz
        tar zxvf vcc2018_submitted_systems_converted_speech.tar.gz
        rm -f vcc2018_submitted_systems_converted_speech.tar.gz
        mv mnt/sysope/test_files/testVCC2/*.wav .
        rm -rf mnt/
        echo "Successfully finished download."
        touch ./.done
    else
        echo "Already exists. Skip download."
    fi
    cd ../
else
    echo "Dataset not supported."
fi