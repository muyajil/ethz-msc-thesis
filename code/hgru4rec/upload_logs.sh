#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No argument supplied"
    exit
fi
for path in $(find . -type f -name \* -print)
do
    target_base="gs://ma-muy/05_logs/$1"
    target_suffix=$(echo $path |  cut -c 2-)
    target=$target_base$target_suffix
    echo "Copying $path to $target"
    gsutil -m cp $path $target
done