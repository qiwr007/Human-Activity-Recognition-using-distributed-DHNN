#1 /bin/bash
shopt -s nullglob
for dir in ~/torch/install/share/lua/5.1/*
do
   #echo $dir
   d=$(echo $dir | grep -Po '[^/]+$')
   #echo $d
   ln -s $dir ~/.luarocks/share/lua/5.1/$d
done

