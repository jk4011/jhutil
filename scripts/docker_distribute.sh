#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

# command example: 
# ------------------------------------------------------------------------------------------
# docker_distribute.sh [host] [container] [tag] [client1 client2 client3]
# docker_distribute.sh node2 part_assembly latest node4 node6

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
echo "docker run example:"
echo ""

echo "nvidia-docker run --name $container \
  -v /data/wlsgur4011:/data/wlsgur4011 \
  -p 23444:23444 \
  -p 88:34555 \
  -e TZ=Asia/Seoul \
  --ipc host \
  --restart unless-stopped \
  -it $image bash" 
