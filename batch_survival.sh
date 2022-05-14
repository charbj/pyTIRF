for i in run_*; do 
	cd $i; 
		for j in pos*; do 
			n=$(echo $j | sed -e 's/pos_/analysis_/'); 
			cd $n;
			python3 ~/projects/MAC/TIRF/pyTIRF/view.py --survival -l ../$j -o ../../$i'_'$j;
			cd ../; 
		done; 
	cd ../; 
done
