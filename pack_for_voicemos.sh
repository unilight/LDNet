#!/usr/bin/env bash

zip_name=$1
main_answer=$2

set -e

# check arguments
if [ $# -le 1 ]; then
    echo "Usage: $0 <zip_name> <main_answer> [<ood_answer>]"
    exit 1
fi

# make dir for output zip file
mkdir -p $(dirname ${zip_name})

# make temp dir
TEMP_DIR=`mktemp -d`

# dump main track answer file
cat ${main_answer} > ${TEMP_DIR}/answer.txt

echo "Main track answer file: ${main_answer}"

# dump optional OOD track answer file
if [ $# -ge 3 ]; then
  ood_answer=$3
  cat ${ood_answer} >> ${TEMP_DIR}/answer.txt
  echo "OOD  track answer file: ${ood_answer}"
fi
echo "=== packing ..."

# compress in zip format
zip -j ${zip_name} ${TEMP_DIR}/answer.txt
echo "=== packing done"
echo "Zipped file: ${zip_name}"

# remove temp dir
rm -rf ${TEMP_DIR}