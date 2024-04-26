#!/bin/bash
set -e        # exit when error
set -o xtrace # print command


math_problem() {
    # two random number 
    a=$(($RANDOM % 187))
    b=$(($RANDOM % 187))
    echo "$a x $b = ?"
    read answer
    if [ $answer -eq $(($a * $b)) ]; then
        echo "Correct!"
    else
        exit 1
    fi
}

allow_youtube() {
    sudo sed -i '' '/youtube/d' /etc/hosts
}

block_youtube() {
    echo "127.0.0.1 youtube.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 www.youtube.com" | sudo tee -a  /etc/hosts
    echo "127.0.0.1 *.youtube.com" | sudo tee -a  /etc/hosts
    dscacheutil -flushcache
    exit 2
}

math_problem

trap block_youtube SIGINT
allow_youtube
sleep 60
block_youtube

