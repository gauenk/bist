#!/bin/bash
./bin/bist -d data/examples/hummingbird/imgs/ -f data/examples/hummingbird/flows/ -o results/hummingbird/ -n 25 --iperc_coeff 4.0 --thresh_new 0.01 --read_video 1 --img_ext png
./bin/bist -d data/examples/kid-football/imgs/ -f data/examples/kid-football/flows/ -o results/kid-football/ -n 25 --iperc_coeff 4.0 --thresh_new 0.05 --read_video 1 --img_ext jpg