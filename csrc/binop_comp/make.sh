#! /bin/bash
make

if [ -s ../../binop_comp ]
then
    mv binop_comp/__init__.py ../../binop_comp
    mv binop_comp/_binop_comp.so ../../binop_comp
else
    mv binop_comp ../../binop_comp
fi
rm -rf binop_comp
make clean