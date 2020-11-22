#!/bin/awk

BEGIN {
    num_algs = 0
}

/Selected algorithm: .*/ {
    alg = $3
    algs[num_algs++] = $3
}

/Preparations.*/ {
    results[alg "prep"] = $3
}

/Execution.*/ {
    results[alg "exec"] = $3
}

/Finalization.*/ {
    results[alg "fin"] = $3
}

END {
    units = " ms"
    print "| Algorithm | Preparations | Execution | Finalization |"
    print "| --------- | ------------ | --------- | ------------ |"
    for (i = 0; i < num_algs; i++) {
        print "| " algs[i] " | " results[algs[i] "prep"] units " | " results[algs[i] "exec"] units " | " results[algs[i] "fin"] units " |"
    }
}