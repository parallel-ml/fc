#!/bin/bash

REPEAT=1

GENERAL="cpu-cycles,bus-cycles,instructions"
BRANCH="branches,branch-misses"
CACHE="cache-references,cache-misses"
CACHE_L1="L1-dcache-load-misses,L1-dcache-loads,L1-dcache-store-misses,L1-dcache-stores,L1-icache-load-misses,L1-icache-loads"
CACHE_LLC="LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores"
OS="page-faults,dTLB-load-misses,dTLB-store-misses,iTLB-load-misses"

ARM='"armv7_cortex_a7/br_immed_retired/","armv7_cortex_a7/br_mis_pred/","armv7_cortex_a7/br_pred/","armv7_cortex_a7/br_return_retired/","armv7_cortex_a7/bus_access/","armv7_cortex_a7/bus_cycles/","armv7_cortex_a7/cid_write_retired/","armv7_cortex_a7/cpu_cycles/","armv7_cortex_a7/exc_return/","armv7_cortex_a7/exc_taken/","armv7_cortex_a7/inst_retired/","armv7_cortex_a7/inst_spec/","armv7_cortex_a7/l1d_cache/","armv7_cortex_a7/l1d_cache_refill/","armv7_cortex_a7/l1d_cache_wb/","armv7_cortex_a7/l1d_tlb_refill/","armv7_cortex_a7/l1i_cache/","armv7_cortex_a7/l1i_cache_refill/","armv7_cortex_a7/l1i_tlb_refill/","armv7_cortex_a7/l2d_cache/","armv7_cortex_a7/l2d_cache_refill/","armv7_cortex_a7/l2d_cache_wb/","armv7_cortex_a7/ld_retired/","armv7_cortex_a7/mem_access/","armv7_cortex_a7/memory_error/","armv7_cortex_a7/pc_write_retired/","armv7_cortex_a7/st_retired/","armv7_cortex_a7/sw_incr/","armv7_cortex_a7/ttbr_write_retired/,"armv7_cortex_a7/unaligned_ldst_retired/"'

#limits=( 10 20 40 60 80 100 120 160 200 220 240 250 320 400 440 480 500) 
limits=( 4 8 16 32 64 128 256 512 1024) 

for i in "${limits[@]}"
do
  echo $i
  #sudo /home/pi/perf/perf  stat -e $GENERAL,$BRANCH,$CACHE,$CAHCE_L1,$CACHE_LLC,$OS,$ARM -r $REPEAT -a ./perf_cnn_layer.py $i l 2>&1 | tee stat_cnn_$i 
  #sudo /home/pi/perf/perf  stat -e $GENERAL,$BRANCH,$CACHE,$CAHCE_L1,$CACHE_LLC,$OS,$ARM -r $REPEAT -a ./perf_cnn_layer.py $i n 2>&1 | tee -a stat_cnn_$i
  ./perf_cnn_layer.py $i n 2>&1 | tee time_cnn_$i
done

