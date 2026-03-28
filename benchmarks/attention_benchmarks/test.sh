python benchmark.py \
    --backends FLASH_ATTN TRITON_ATTN FLASHINFER \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --output-csv results_normal.csv

python benchmark.py \
     --backends CUTLASS_MLA \
     FLASHINFER_MLA --batch-specs "64q1s1k" \
     --output-csv results_mla.csv