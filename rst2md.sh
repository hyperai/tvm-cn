#!/bin/bash

# This script was created to convert a directory full of rst files into
# markdown equivalents. It uses pandoc to do the conversion.
#
# 1. Install pandoc from https://pandoc.org/
# 2. Run the script
#
# By default this will keep the original .rst file
#
# Original by https://gist.github.com/stupergenius/b43d38fbff0547c96322e3f44264a969
# Modified by Tunghsiao Liu <t@sparanoid.com>

set -o errexit
set -o nounset
set -o pipefail

_main() {
  path=$1
  for file in $(find $path -name '*.rst'); do
    local pwd=$(pwd)
    local filename_md=$(echo $file | sed "s/\.rst$/\.md/g")
    local filename_orig=$(echo $file | sed "s|$pwd|.|g")
    echo "Converting $filename_orig"
    `pandoc -s -o "${filename_md}" ${file}`
    # uncomment this line to delete the source file.
    # rm "${file}"
  done
}

main() {
  oldifs=$IFS
  IFS=$'\n'

  _main ${1:-$(pwd)}

  IFS=$oldifs
}

main $@

# do
#   filename="${f%.*}"
#   echo "Converting $f to $filename.md"
#   `pandoc -s -o $filename.md $f`
#   # uncomment this line to delete the source file.
#   # rm $f
# done
