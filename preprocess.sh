for i in run_*; do 
	cd $i; 
		for j in pos*; do 
			n=$(echo $j | sed -e 's/pos_/analysis_/'); 
			cd $n; 
			python3 ~/projects/MAC/TIRF/pyTIRF/view.py -i ../$j -g ~/projects/MAC/TIRF/220428/gain_ref.tiff --batch; 
			cd ../; 
		done; 
	cd ../; 
done
