#!/bin/bash
for i in {1..500}
do
    start=`date +%s.%N`
    # extract essential data from raw data
    sudo ovs-ofctl dump-flows s1 > data/raw1.txt
    grep "nw_src" data/raw1.txt > data/flowentries1.csv
    ipsrc=$(awk -F "," '{out=""; for(k=2;k<=NF;k++){out=out" "$k}; print out}' data/flowentries1.csv | awk -F " " '{split($14,d,"="); print d[2]","}')
    ipdst=$(awk -F "," '{out=""; for(k=2;k<=NF;k++){out=out" "$k}; print out}' data/flowentries1.csv | awk -F " " '{split($15,d,"="); print d[2]","}')
    inport=$(awk -F "," '{out=""; for(k=2;k<=NF;k++){out=out" "$k}; print out}' data/flowentries1.csv | awk -F " " '{split($10,d,"="); print d[2]","}')
    # check if there are no traffics in the network at the moment.
    if test -z "$ipsrc" || test -z "$ipdst" || test -z "$inport" 
    then
        state=0
    else
        echo "$ipsrc" > data/ipsrc1.csv
        echo "$ipdst" > data/ipdst1.csv
        echo "$inport" > data/inport1.csv
        echo "Collect xong lan $i"
        
    fi
    sleep 1
    # echo "Ket qua lan $(($i - 1))"
    echo "Ket qua lan $i:"
    state=$(awk '{print $0;}' result.txt)
    if [ $state -eq 1 ];
        then
            echo "Network is under attack"
        else
            echo "Network is normal"
    fi
    echo "------Doi lan tiep theo-------" 
    echo " "
    end=`date +%s.%N`
    runtime=$( echo "$end - $start" | bc -l )
    timesleep=$( echo "3 - $runtime" | bc -l )
    sleep $timesleep
done



# ==============================================================================================================================================
# Ref
# Get all fields (n columns) in awk: https://stackoverflow.com/a/2961711/11806074
# e.g. awk -F "," '{out=""; for(i=2;i<=NF;i++){out=out" "$i" "i}; print out}' data/flowentries.csv 

# ovs-ofctl reference
# add-flow SWITCH FLOW        add flow described by FLOW    e.g. ... add-flow s1 "flow info"
# add-flows SWITCH FILE       add flows from FILE           e.g. ... add-flows s1 flows.txt

# example of multiple commands in awk, these commands below extract ip_src and ip_dst from flow entries
# awk -F "," '{split($10,c,"="); print c[2]","}' data/flowentries.csv > data/ipsrc.csv
# awk -F "," '{split($11,d,"=");  split(d[2],e," "); print e[1]","}' data/flowentries.csv > data/ipdst.csv