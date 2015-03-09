for k in 2 3 4 5
do
  for p1 in 1 2 3
  do
    for p2 in 1 2 3
    do
      for v in 1 2 3
      do
        for d in 5 7 10
        do
          condor_submit likelihood.condor \
            -append arguments="likelihood.py $k $p1 $p2 $v $d" \
            -append error="log_likelihood-$k-$p1-$p2-$v-$d.err" \
            -append log="log_likelihood-$k-$p1-$p2-$v-$d.log" \
            -append output="log_likelihood-$k-$p1-$p2-$v-$d.out"
        done
      done
    done
  done
done
