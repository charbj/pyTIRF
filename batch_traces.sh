for i in run_5*; do 
	cd $i; 
		for j in pos*; do 
			n=$(echo $j | sed -e 's/pos_/analysis_/'); 
			cd $n;
			#mkdir traces 
			python3 ~/projects/MAC/TIRF/pyTIRF/view.py --traces -i ch0_stack.tiff -i2 ch1_stack.tiff -l ../$j -o traces/;
			cd ../; 
		done; 
	cd ../; 
done
