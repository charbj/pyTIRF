for i in run_*; do cd $i; for j in pos*; do n=$(echo $j | sed -e 's/pos_/analysis_/'); mkdir $n; done; cd ../; done
