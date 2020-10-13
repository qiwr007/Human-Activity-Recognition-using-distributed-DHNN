#! /bin/bash

if [[ $(pgrep redis) == "" ]] ;
then
   redis-server redis.conf
   echo "Started Redis.  PID = $(pgrep redis)"
fi
#killall redis-server
#redis-server redis.conf

function callRedis {
    redisPass=`grep -Po "(?<=requirepass ).*$" redis.conf`
    redis-cli -h 127.0.0.1 -a $redisPass $*
}

# might be better to run this from another machine
function issueStartCommand {
    callRedis PUBLISH meta init
}

function monitorRedis {
    callRedis MONITOR >> monitor.log
}

if [[ $(hostname) == "viz12" || $(which th) == $HOME/torch/install/bin/th ]] ;
then
   rm g*.txt # w files will get clobbered by cp on next line
   rm client/w*.txt #Clear out old initial weights
   rm server/w*.txt #Clear out old initial weights
   th generate_weights.lua
   callRedis FLUSHDB
   th init.lua #&
   #TORCH_PID=$!
   #sleep 3
   #issueStartCommand
   #echo "PID of global Torch process is $TORCH_PID."
else
   echo "Please install Torch locally (in ~/torch) and ensure that the local \
th/luarocks/lua/etc. executables are first in the path (i.e., do not load the \
Torch module).  This is needed because we cannot install our dependencies in \
the global version as easily as we can in the local version."
fi
