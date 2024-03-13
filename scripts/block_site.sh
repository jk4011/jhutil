#!/bin/bash
set -e        # exit when error
set -o xtrace # print command


block_site() {
    local addr=$1
    # if addr is empty, exit
    if [ -z "$addr" ]; then
        echo "No address provided. Exiting."
        exit 1
    fi
    echo "127.0.0.1 ${addr}" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 www.${addr}" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 *.${addr}" | sudo tee -a  /etc/hosts
    exit 2
}

block_site "$1"