#!/usr/bin/sh

clear
make clean
make -j4 all
./bin/program < input.in
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./bin/program < input.in
