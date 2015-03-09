for N in 10000 20000 50000 
  do
    for K in 333 1000
      do
        for E in 500 1000 5000
          do
            condor_submit experiment_series.condor \
              -append arguments="experiment_series.py $N $K $E" \
              -append error="log_series-$N-$K-$E.err" \
              -append log="log_series-$N-$K-$E.log" \
              -append output="log_series-$N-$K-$E.out"
          done
      done
   done
