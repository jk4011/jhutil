#!/bin/bash
set -e
# set -o xtrace # print command

cd $jhutil_path
# print all argument 
python send_slack.py $@

