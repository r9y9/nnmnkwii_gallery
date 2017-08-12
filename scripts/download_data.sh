#!/bin/bash

set -e

name=$1

# name can be one of
# 1. slt_arctic_demo_data
# 2. slt_arctic_full_data

question_name="questions-radio_dnn_416.hed"

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
data_root=$script_dir/../data

cd $data_root

base=http://104.131.174.95/

zipname=${name}.zip
dst_dir=$data_root/$name
url=$base$zipname


if [ -d $dst_dir ]; then
    echo "$name already downloaded"
    exit 0
fi

echo "Downloading from $url..."
curl -L -o $zipname $url
unzip -q -o $zipname
# Arange structure and remove unnecessary files
mv $dst_dir/merlin_baseline_practice/acoustic_data/label_state_align $dst_dir
mv $dst_dir/merlin_baseline_practice/acoustic_data/label_phone_align $dst_dir
ln -sf $data_root/$question_name $dst_dir/$question_name
rm -rf $dst_dir/lab $dst_dir/merlin_baseline_practice $dst_dir/cmuarctic.data

rm -rf $zipname
