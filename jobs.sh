#!/bin/bash
python new_train_network.py -at 0 -bs 300 -ne 300 -lr 0.0005 -wd 0.001 -nc 2 -nf 32 64 -fs 11 9 -nfc 4 -nn 1024 1024 1024 1024
python new_train_network.py -at 1 -bs 300 -ne 300 -lr 0.0005 -wd 0.001 -nc 2 -nf 32 64 -fs 11 9 -nfc 4 -nn 1024 1024 1024 1024
python new_train_network.py -at 2 -bs 300 -ne 300 -lr 0.0005 -wd 0.001 -nc 2 -nf 32 64 -fs 11 9 -nfc 4 -nn 1024 1024 1024 1024


