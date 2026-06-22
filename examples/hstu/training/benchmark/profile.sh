export PPU_LIB_PERF_INSTRUMENT=1  #asys profile时将对应的ut string标记在profile结果上
export PPU_FMHA_PERF_INSTRUMENT=1
export ACEXT_SHOW_TOKEN_EXPERT_DIST=1 # acext MOE新增

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 #设置在8卡或者16卡上运行


#cd /ppuswhome/longjie.long/partner/tests/Wan/DiffSynth-Studio/

#采集系统内所有进程的CPU调度跟踪数据，采集Python函数，内存使用情况跟踪
asys profile -s system-wide \
--hggc-memory-usage=true --host-memory-sampling=true  \
-t hggc,hgtx,acblas,acdnn,osrt  \
--python-functions-trace all --python-backtrace=hggc --python-backtrace-depth=50  \
--capture-range hggcProfilerApi  --capture-range-end=stop-shutdown  \
-o  ./logs/h20-`date +"%m%d-%H%M"` \
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh  \
2>&1 | tee ./logs/h20-`date +"%m%d-%H%M"`-asys.log

