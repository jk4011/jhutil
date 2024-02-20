#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

sudo sed -i '' '/google/d' /etc/hosts
sleep 60

#TODO: 이거 실행 어렵게 하기 위한, 수학 문제 만들기

echo "127.0.0.1 google.com" | sudo tee -a  /etc/hosts
echo "127.0.0.1 www.google.com" | sudo tee -a  /etc/hosts
echo "127.0.0.1 *.google.com" | sudo tee -a  /etc/hosts
