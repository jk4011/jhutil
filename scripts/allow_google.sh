#!/bin/bash
set -e        # exit when error
set -o xtrace # print command


math_problem() {
    # two random number 
    a=$(($RANDOM % 987))
    b=$(($RANDOM % 987))
    echo "$a x $b = ?"
    read answer
    if [ $answer -eq $(($a * $b)) ]; then
        echo "Correct!"
    else
        exit 1
    fi
}

allow_google() {
    sudo sed -i '' '/google/d' /etc/hosts
}

block_google() {
    echo "127.0.0.1 google.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 www.google.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 *.google.com" | sudo tee -a  /etc/hosts
    dscacheutil -flushcache
    exit 2
}

math_problem

trap block_google SIGINT
allow_google
sleep 60
block_google

