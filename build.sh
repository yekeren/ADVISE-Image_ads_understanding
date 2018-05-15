#!/bin/sh

set -x

protoc protos/*.proto --python_out=. || exit -1
find ./ -name "*.pyc" | xargs rm -f

exit 0
