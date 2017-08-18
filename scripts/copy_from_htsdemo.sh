#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
data_root=$script_dir/../data

cd $data_root

HTS_DEMO_ROOT=$1

DST_DIR=$data_root/"NIT-ATR503"
if [ -d $DST_DIR ]; then
    echo "`basename $HTS_DEMO_ROOT` already copied"
    exit 0
fi
mkdir -p $DST_DIR

# Full context Labels
if [ -d $DST_DIR/label_phone_align ]; then
    rm -rf $DST_DIR/label_phone_align
fi
cp -r $HTS_DEMO_ROOT/data/labels/full $DST_DIR/label_phone_align

# Test full context labels
if [ -d $DST_DIR/test_label_phone_align ]; then
    rm -rf $DST_DIR/test_label_phone_align
fi
cp -r $HTS_DEMO_ROOT/data/labels/gen $DST_DIR/test_label_phone_align

# Wav files
if [ -d $DST_DIR/wav ]; then
    rm -rf $DST_DIR/wav
fi
cp -r $HTS_DEMO_ROOT/data/raw $DST_DIR/wav
cd $DST_DIR/wav
for f in *.raw
do
    raw2wav -s 48.0 $f
    rm -f $f
done
