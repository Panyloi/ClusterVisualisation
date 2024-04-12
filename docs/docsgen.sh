#!/bin/bash

if [ $# != 1 ]; then
    echo "Usage:"
    echo "docsgen.sh [python interpreter path with pylint]"
    exit 1
fi

SDIR=$(dirname "$0")

# run python script on the input interpreter which runs pyreverse
$1 -c "from pylint import *; run_pyreverse([\"$SDIR/../src\", \"-d\", \"$SDIR\", \"-o\", \"dot\"])"

# generate charts
dot -Tpng $SDIR/classes.dot -o $SDIR/classes.png
dot -Tpng $SDIR/packages.dot -o $SDIR/packages.png

rm $SDIR/classes.dot
rm $SDIR/packages.dot

exit 0
