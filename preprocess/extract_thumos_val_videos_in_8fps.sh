avi=`find thumos_video_val -name \*.mp4`  #val

detection=`cat annotations/val/*.txt|cut -d' ' -f1 | sort | uniq`

for i in $avi; do  
  dir=`echo $i | cut -d. -f1` 
  f1=`echo $dir | cut -d/ -f2` 
  out='./data/thumos/video_8fps/validation/'$f1'.mp4'
  echo $out
  ffmpeg -i $i -r 8 -strict -2 $out
done