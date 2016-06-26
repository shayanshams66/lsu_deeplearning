*** Error in `/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/lt-singa': munmap_chunk(): invalid pointer
: 0x000000000278c2e8 ***
======= Backtrace: =========
/lib64/libc.so.6(+0x7b194)[0x7f0cc47dc194]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZN5singa9LstmLayer14ComputeFeatureEiRKSt6
vectorIPNS_5LayerESaIS3_EE+0x886)[0x7f0cc8d909e8]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZN5singa10BPTTWorker7ForwardEiNS_5PhaseEP
NS_9NeuralNetE+0x5d1)[0x7f0cc8ccc9c7]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZN5singa8BPWorker13TrainOneBatchEiPNS_9Ne
uralNetE+0x36)[0x7f0cc8ccb7a0]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZN5singa6Worker3RunEv+0x721)[0x7f0cc8cc91
bb]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZNKSt7_Mem_fnIMN5singa6WorkerEFvvEEclIJEv
EEvPS1_DpOT_+0x65)[0x7f0cc8cbba25]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZNSt12_Bind_simpleIFSt7_Mem_fnIMN5singa6W
orkerEFvvEEPS2_EE9_M_invokeIJLm0EEEEvSt12_Index_tupleIJXspT_EEE+0x43)[0x7f0cc8cbb87f]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZNSt12_Bind_simpleIFSt7_Mem_fnIMN5singa6W
orkerEFvvEEPS2_EEclEv+0x1b)[0x7f0cc8cbb71f]
/home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0(_ZNSt6thread5_ImplISt12_Bind_simpleIFSt7_M
em_fnIMN5singa6WorkerEFvvEEPS4_EEE6_M_runEv+0x1c)[0x7f0cc8cbb650]
/lib64/libstdc++.so.6(+0xb5220)[0x7f0cc50f0220]
/lib64/libpthread.so.0(+0x7dc5)[0x7f0cc7760dc5]
/lib64/libc.so.6(clone+0x6d)[0x7f0cc4857ced]
======= Memory map: ========
00400000-00402000 r-xp 00000000 fd:02 1262241459                         /home/shayanshams/Documents/singa/singa_lstm/incubator-si
nga-master/.libs/lt-singa
00601000-00602000 r--p 00001000 fd:02 1262241459                         /home/shayanshams/Documents/singa/singa_lstm/incubator-si
nga-master/.libs/lt-singa
00602000-00603000 rw-p 00002000 fd:02 1262241459                         /home/shayanshams/Documents/singa/singa_lstm/incubator-si
nga-master/.libs/lt-singa
02532000-02a05000 rw-p 00000000 00:00 0                                  [heap]
7f0cb4000000-7f0cb43af000 rw-p 00000000 00:00 0
7f0cb43af000-7f0cb8000000 ---p 00000000 00:00 0
7f0cbc000000-7f0cbc021000 rw-p 00000000 00:00 0
7f0cbc021000-7f0cc0000000 ---p 00000000 00:00 0
7f0cc0afa000-7f0cc2afa000 rw-p 00000000 00:00 0
7f0cc2afa000-7f0cc3201000 rw-p 00000000 00:00 0
7f0cc3201000-7f0cc3202000 ---p 00000000 00:00 0
7f0cc3202000-7f0cc3a02000 rw-p 00000000 00:00 0                          [stack:31871]
7f0cc3a02000-7f0cc3a03000 ---p 00000000 00:00 0
7f0cc3a03000-7f0cc4203000 rw-p 00000000 00:00 0                          [stack:31870]
7f0cc4203000-7f0cc423e000 r-xp 00000000 fd:01 202997051                  /usr/lib64/libquadmath.so.0.0.0
7f0cc423e000-7f0cc443d000 ---p 0003b000 fd:01 202997051                  /usr/lib64/libquadmath.so.0.0.0
7f0cc443d000-7f0cc443e000 r--p 0003a000 fd:01 202997051                  /usr/lib64/libquadmath.so.0.0.0
7f0cc443e000-7f0cc443f000 rw-p 0003b000 fd:01 202997051                  /usr/lib64/libquadmath.so.0.0.0
7f0cc443f000-7f0cc455e000 r-xp 00000000 fd:01 214353374                  /usr/lib64/libgfortran.so.3.0.0
7f0cc455e000-7f0cc475e000 ---p 0011f000 fd:01 214353374                  /usr/lib64/libgfortran.so.3.0.0
7f0cc475e000-7f0cc475f000 r--p 0011f000 fd:01 214353374                  /usr/lib64/libgfortran.so.3.0.0
7f0cc475f000-7f0cc4761000 rw-p 00120000 fd:01 214353374                  /usr/lib64/libgfortran.so.3.0.0
7f0cc4761000-7f0cc4918000 r-xp 00000000 fd:01 201327704                  /usr/lib64/libc-2.17.so
7f0cc4918000-7f0cc4b18000 ---p 001b7000 fd:01 201327704                  /usr/lib64/libc-2.17.so
7f0cc4b18000-7f0cc4b1c000 r--p 001b7000 fd:01 201327704                  /usr/lib64/libc-2.17.so
7f0cc4b1c000-7f0cc4b1e000 rw-p 001bb000 fd:01 201327704                  /usr/lib64/libc-2.17.so
7f0cc4b1e000-7f0cc4b23000 rw-p 00000000 00:00 0
7f0cc4b23000-7f0cc4b38000 r-xp 00000000 fd:01 214293693                  /usr/lib64/libgcc_s-4.8.5-20150702.so.1
7f0cc4b38000-7f0cc4d37000 ---p 00015000 fd:01 214293693                  /usr/lib64/libgcc_s-4.8.5-20150702.so.1
7f0cc4d37000-7f0cc4d38000 r--p 00014000 fd:01 214293693                  /usr/lib64/libgcc_s-4.8.5-20150702.so.1
7f0cc4d38000-7f0cc4d39000 rw-p 00015000 fd:01 214293693                  /usr/lib64/libgcc_s-4.8.5-20150702.so.1
7f0cc4d39000-7f0cc4e3a000 r-xp 00000000 fd:01 203236988                  /usr/lib64/libm-2.17.so
7f0cc4e3a000-7f0cc5039000 ---p 00101000 fd:01 203236988                  /usr/lib64/libm-2.17.so
7f0cc5039000-7f0cc503a000 r--p 00100000 fd:01 203236988                  /usr/lib64/libm-2.17.so
7f0cc503a000-7f0cc503b000 rw-p 00101000 fd:01 203236988                  /usr/lib64/libm-2.17.so
7f0cc503b000-7f0cc5124000 r-xp 00000000 fd:01 205727150                  /usr/lib64/libstdc++.so.6.0.19
7f0cc5124000-7f0cc5324000 ---p 000e9000 fd:01 205727150                  /usr/lib64/libstdc++.so.6.0.19
7f0cc5324000-7f0cc532c000 r--p 000e9000 fd:01 205727150                  /usr/lib64/libstdc++.so.6.0.19
7f0cc532c000-7f0cc532e000 rw-p 000f1000 fd:01 205727150                  /usr/lib64/libstdc++.so.6.0.19
7f0cc532e000-7f0cc5343000 rw-p 00000000 00:00 0
7f0cc5343000-7f0cc7532000 r-xp 00000000 fd:01 203106258                  /usr/lib64/libopenblas-r0.2.16.so
7f0cc7532000-7f0cc7732000 ---p 021ef000 fd:01 203106258                  /usr/lib64/libopenblas-r0.2.16.so
7f0cc7732000-7f0cc7736000 r--p 021ef000 fd:01 203106258                  /usr/lib64/libopenblas-r0.2.16.so
7f0cc7736000-7f0cc774f000 rw-p 021f3000 fd:01 203106258                  /usr/lib64/libopenblas-r0.2.16.so
7f0cc774f000-7f0cc7759000 rw-p 00000000 00:00 0
7f0cc7759000-7f0cc776f000 r-xp 00000000 fd:01 201398107                  /usr/lib64/libpthread-2.17.so
7f0cc776f000-7f0cc796f000 ---p 00016000 fd:01 201398107                  /usr/lib64/libpthread-2.17.so
7f0cc796f000-7f0cc7970000 r--p 00016000 fd:01 201398107                  /usr/lib64/libpthread-2.17.so
7f0cc7970000-7f0cc7971000 rw-p 00017000 fd:01 201398107                  /usr/lib64/libpthread-2.17.so
7f0cc7971000-7f0cc7975000 rw-p 00000000 00:00 0
7f0cc7975000-7f0cc7979000 r-xp 00000000 fd:01 203034758                  /usr/lib64/libuuid.so.1.3.0
7f0cc7979000-7f0cc7b78000 ---p 00004000 fd:01 203034758                  /usr/lib64/libuuid.so.1.3.0
7f0cc7b78000-7f0cc7b79000 r--p 00003000 fd:01 203034758                  /usr/lib64/libuuid.so.1.3.0
7f0cc7b79000-7f0cc7b7a000 rw-p 00004000 fd:01 203034758                  /usr/lib64/libuuid.so.1.3.0
7f0cc7b7a000-7f0cc7b81000 r-xp 00000000 fd:01 203248936                  /usr/lib64/librt-2.17.so
7f0cc7b81000-7f0cc7d80000 ---p 00007000 fd:01 203248936                  /usr/lib64/librt-2.17.so
7f0cc7d80000-7f0cc7d81000 r--p 00006000 fd:01 203248936                  /usr/lib64/librt-2.17.so
7f0cc7d81000-7f0cc7d82000 rw-p 00007000 fd:01 203248936                  /usr/lib64/librt-2.17.so
7f0cc7d82000-7f0cc7dc1000 r-xp 00000000 fd:01 68011751                   /usr/local/lib/libzmq.so.3.0.0
7f0cc7dc1000-7f0cc7fc1000 ---p 0003f000 fd:01 68011751                   /usr/local/lib/libzmq.so.3.0.0
7f0cc7fc1000-7f0cc7fc6000 r--p 0003f000 fd:01 68011751                   /usr/local/lib/libzmq.so.3.0.0
7f0cc7fc6000-7f0cc7fc7000 rw-p 00044000 fd:01 68011751                   /usr/local/lib/libzmq.so.3.0.0
7f0cc7fc7000-7f0cc8039000 r-xp 00000000 fd:01 68011759                   /usr/local/lib/libczmq.so.1.1.0
7f0cc8039000-7f0cc8239000 ---p 00072000 fd:01 68011759                   /usr/local/lib/libczmq.so.1.1.0
7f0cc8239000-7f0cc823a000 r--p 00072000 fd:01 68011759                   /usr/local/lib/libczmq.so.1.1.0
7f0cc823a000-7f0cc823c000 rw-p 00073000 fd:01 68011759                   /usr/local/lib/libczmq.so.1.1.0
7f0cc823c000-7f0cc8259000 r-xp 00000000 fd:01 68257474                   /usr/local/lib/libglog.so.0.0.0
7f0cc8259000-7f0cc8459000 ---p 0001d000 fd:01 68257474                   /usr/local/lib/libglog.so.0.0.0
7f0cc8459000-7f0cc845a000 r--p 0001d000 fd:01 68257474                   /usr/local/lib/libglog.so.0.0.0
7f0cc845a000-7f0cc845b000 rw-p 0001e000 fd:01 68257474                   /usr/local/lib/libglog.so.0.0.0
7f0cc845b000-7f0cc846b000 rw-p 00000000 00:00 0
7f0cc846b000-7f0cc8480000 r-xp 00000000 fd:01 201398180                  /usr/lib64/libz.so.1.2.7
7f0cc8480000-7f0cc867f000 ---p 00015000 fd:01 201398180                  /usr/lib64/libz.so.1.2.7
7f0cc867f000-7f0cc8680000 r--p 00014000 fd:01 201398180                  /usr/lib64/libz.so.1.2.7
7f0cc8680000-7f0cc8681000 rw-p 00015000 fd:01 201398180                  /usr/lib64/libz.so.1.2.7
7f0cc8681000-7f0cc878c000 r-xp 00000000 fd:01 68527141                   /usr/local/lib/libprotobuf.so.9.0.0
7f0cc878c000-7f0cc898b000 ---p 0010b000 fd:01 68527141                   /usr/local/lib/libprotobuf.so.9.0.0
7f0cc898b000-7f0cc898f000 r--p 0010a000 fd:01 68527141                   /usr/local/lib/libprotobuf.so.9.0.0
7f0cc898f000-7f0cc8992000 rw-p 0010e000 fd:01 68527141                   /usr/local/lib/libprotobuf.so.9.0.0
7f0cc8992000-7f0cc8e32000 r-xp 00000000 fd:02 1255988561                 /home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0.0.0
7f0cc8e32000-7f0cc9031000 ---p 004a0000 fd:02 1255988561                 /home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0.0.0
7f0cc9031000-7f0cc9046000 r--p 0049f000 fd:02 1255988561                 /home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0.0.0
7f0cc9046000-7f0cc905c000 rw-p 004b4000 fd:02 1255988561                 /home/shayanshams/Documents/singa/singa_lstm/incubator-singa-master/.libs/libsinga.so.0.0.0
7f0cc905c000-7f0cc905d000 rw-p 00000000 00:00 0
7f0cc905d000-7f0cc907e000 r-xp 00000000 fd:01 203236982                  /usr/lib64/ld-2.17.so
7f0cc9153000-7f0cc9260000 rw-p 00000000 00:00 0
7f0cc9277000-7f0cc927e000 rw-p 00000000 00:00 0
7f0cc927e000-7f0cc927f000 r--p 00021000 fd:01 203236982                  /usr/lib64/ld-2.17.so
7f0cc927f000-7f0cc9280000 rw-p 00022000 fd:01 203236982                  /usr/lib64/ld-2.17.so
7f0cc9280000-7f0cc9281000 rw-p 00000000 00:00 0
7fff35957000-7fff35978000 rw-p 00000000 00:00 0                          [stack]
7fff3597a000-7fff3597c000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]
./bin/singa-run.sh: line 107: 31855 Aborted
