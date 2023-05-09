#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

# command example: 
# ------------------------------------------------------------------------------------------
# docker_distribute.sh node6 wlsgur4011/imc node4 node5
# ------------------------------------------------------------------------------------------

host=$1
image=$2
clients=("${@:3}")
file_path=/data/wlsgur4011/docker/images/${2//\//:}.tar
echo "saving $file_path..."
ssh $host docker save $image -o $file_path
echo "docker save $image -o $file_path"
# ssh $host chmod 777 $file_path
echo "saving done!"

for client in "${clients[@]}"
do
    echo "loading $file_path to $client..."
    ssh $client docker load -i $file_path &
done

send_slack.sh "`docker_distribute.sh` $host $image ${clients[@]} done!"