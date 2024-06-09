avi=`find data/thumos_video_test -name \*.mp4`  #test

for i in $avi; do
  dir=`echo $i | cut -d. -f1`
  echo $dir
  f1=`echo  $dir | cut -d/ -f3`
  echo $f1
  out='./data/thumos/video_8fps/test/'$f1'.mp4'
  echo $out
  ffmpeg -i $i -r 8 -strict -2 $out
done