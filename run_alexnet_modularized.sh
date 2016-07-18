#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python alexnet_modularized.py >> $1
