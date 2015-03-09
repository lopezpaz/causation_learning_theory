for K in 333 
  do
    for E in 100 250 500 1000 5000
      do
      for L in 5 10 20 50 100 
        do
          condor_submit experiment_challenge.condor \
            -append arguments="experiment_challenge.py $K $E $L" \
            -append error="log_challenge2-$K-$E-$L.err" \
            -append log="log_challenge2-$K-$E-$L.log" \
            -append output="log_challenge2-$K-$E-$L.out"
        done
    done
done
