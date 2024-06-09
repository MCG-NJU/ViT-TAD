avi=`find data/thumos_video_val -name \*.mp4`  #val

for i in $avi; do
  dir=`echo $i | cut -d. -f1`
  f1=`echo $dir | cut -d/ -f3`
  out='./data/thumos/video_8fps/validation/'$f1'.mp4'
  echo $out
  ffmpeg -i $i -r 8 -strict -2 $out
done