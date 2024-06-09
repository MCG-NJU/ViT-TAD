avi=`find thumos_video_test -name \*.mp4`  #test

detection=`cat annotations/test/*.txt|cut -d' ' -f1 | sort | uniq`

for i in $avi; do  
  dir=`echo $i | cut -d. -f1` 
  f1=`echo $dir | cut -d/ -f2` 
  out='./data/thumos/video_8fps/test/'$f1'.mp4'
  echo $out
  ffmpeg -i $i -r 8 -strict -2 $out
done