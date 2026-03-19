# profile handwritten triton kernel （gui version）
ncu -f \
    --set full \
    --kernel-name "regex:_rms_norm_kernel" \
    --launch-skip 5 --launch-count 3 \
    --target-processes all \
    -o rmsnorm_triton_profile \
    python benchmark_rmsnorm.py

# profile handwritten triton kernel （terminal）
ncu \
    --set full \
    --kernel-name "regex:_rms_norm_kernel" \
    --launch-skip 5 --launch-count 3 \
    --target-processes all \
    python benchmark_rmsnorm.py \
    2>&1 | tee ncu_rep.log