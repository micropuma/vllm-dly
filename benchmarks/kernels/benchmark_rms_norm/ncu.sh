#!/bin/bash
set -euo pipefail

# ===================================================================
# NCU profiling wrapper for benchmark_rmsnorm.py
#
# Usage:
#   bash ncu.sh [gui|terminal] [OPTIONS]
#
# Options:
#   --kernel   triton|inductor|vllm|flashinfer|torch_compile|naive  (default: triton)
#   --bs       batch size              (default: 4)
#   --seq      sequence length         (default: 128)
#   --hidden   hidden size             (default: 4096)
#   --warmup   warmup iterations       (default: 3)
#   --residual enable residual path
#   --output   output report name      (default: auto-generated)
#
# Examples:
#   bash ncu.sh gui --kernel triton --seq 128 --hidden 4096
#   bash ncu.sh terminal --kernel vllm --bs 1 --seq 512 --hidden 8192 --warmup 5
#   bash ncu.sh gui --kernel inductor --seq 64 --hidden 4096 --residual
# ===================================================================

MODE="${1:-gui}"
shift || true

KERNEL="triton"
BS=4
SEQ=128
HIDDEN=4096
WARMUP=3
RESIDUAL=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)   KERNEL="$2";   shift 2 ;;
        --bs)       BS="$2";       shift 2 ;;
        --seq)      SEQ="$2";      shift 2 ;;
        --hidden)   HIDDEN="$2";   shift 2 ;;
        --warmup)   WARMUP="$2";   shift 2 ;;
        --residual) RESIDUAL="--use-residual"; shift ;;
        --output)   OUTPUT="$2";   shift 2 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

NUM_TOKENS=$((BS * SEQ))
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="ncu_${KERNEL}_${NUM_TOKENS}tok_h${HIDDEN}"
    [[ -n "$RESIDUAL" ]] && OUTPUT="${OUTPUT}_residual"
fi

KERNEL_REGEX_MAP_triton="_rms_norm_kernel"
KERNEL_REGEX_MAP_inductor="triton_"
KERNEL_REGEX_MAP_vllm="rms_norm_kernel"
KERNEL_REGEX_MAP_flashinfer="RMSNorm"
KERNEL_REGEX_MAP_torch_compile="triton_"
KERNEL_REGEX_MAP_naive="native"

REGEX_VAR="KERNEL_REGEX_MAP_${KERNEL}"
KERNEL_REGEX="${!REGEX_VAR:-$KERNEL}"

PY_CMD="python benchmark_rmsnorm.py \
    --ncu-profile ${KERNEL} \
    --batch-size ${BS} \
    --seq-len ${SEQ} \
    --hidden-size ${HIDDEN} \
    --warmup ${WARMUP} \
    ${RESIDUAL}"

echo "============================================"
echo " NCU Profile: ${KERNEL}"
echo " Shape: bs=${BS}, seq=${SEQ}, hidden=${HIDDEN}"
echo " num_tokens: ${NUM_TOKENS}"
echo " warmup: ${WARMUP} launches (skipped by ncu)"
echo " kernel regex: ${KERNEL_REGEX}"
echo " mode: ${MODE}"
echo " output: ${OUTPUT}"
echo "============================================"

NCU_COMMON_ARGS="\
    --set full \
    --kernel-name regex:${KERNEL_REGEX} \
    --launch-skip ${WARMUP} --launch-count 1 \
    --target-processes all"

case "$MODE" in
    gui)
        echo "[GUI mode] generating ${OUTPUT}.ncu-rep ..."
        ncu -f \
            ${NCU_COMMON_ARGS} \
            -o "${OUTPUT}" \
            ${PY_CMD}
        echo "Report saved: ${OUTPUT}.ncu-rep"
        ;;
    terminal)
        echo "[Terminal mode] printing to stdout ..."
        ncu \
            ${NCU_COMMON_ARGS} \
            ${PY_CMD} \
            2>&1 | tee "${OUTPUT}.log"
        echo "Log saved: ${OUTPUT}.log"
        ;;
    dry-run)
        echo "[Dry-run] would execute:"
        echo "  ncu ${NCU_COMMON_ARGS} -o ${OUTPUT} ${PY_CMD}"
        ;;
    *)
        echo "Unknown mode: ${MODE} (use gui|terminal|dry-run)"
        exit 1
        ;;
esac
