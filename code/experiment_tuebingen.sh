for N in 10000
  do
    for K in 333
      do
        for E in 500 1000 5000
          do
            condor_submit experiment_tuebingen.condor \
              -append arguments="experiment_tuebingen.py $N $K $E" \
              -append error="log_tuebingen-$N-$K-$E.err" \
              -append log="log_tuebingen-$N-$K-$E.log" \
              -append output="log_tuebingen-$N-$K-$E.out"
          done
      done
   done
