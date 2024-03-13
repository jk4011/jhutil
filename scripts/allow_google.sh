#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

allow_google() {
    sudo sed -i '' '/google/d' /etc/hosts
}

block_google() {
    echo "127.0.0.1 google.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 www.google.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 *.google.com" | sudo tee -a  /etc/hosts
    exit 2
}

trap block_google SIGINT

allow_google
sleep 60
block_google

