#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

# command example: 
# ------------------------------------------------------------------------------------------
# docker_distribute_hub.sh [host] [container] [tag] [client1 client2 client3]
# docker_distribute_hub.sh node2 part_assembly latest node4 node6

# ------------------------------------------------------------------------------------------

host=$1
container=$2
tag=$3
image=wlsgur4011/$container:$tag
clients=("${@:4}")

ssh $host docker commit $container $image
echo "host commit done!"
ssh $host docker push $image
echo "host push done!"

for client in "${clients[@]}"
do
    ssh $client docker pull $image &
done
wait

echo "distribute done!"
echo ""
