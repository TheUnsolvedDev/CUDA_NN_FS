#!/usr/bin/sh

clear
make clean
make all
./bin/program 100
# valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./bin/program 1