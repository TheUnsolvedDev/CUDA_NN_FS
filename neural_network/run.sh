#!/usr/bin/sh

make clean
make all
# ./bin/program
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./bin/program 5