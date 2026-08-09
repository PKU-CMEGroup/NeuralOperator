#!/bin/bash
#SBATCH -o decimate_%A_%a.out
#SBATCH --qos=low
#SBATCH -J decimate
#SBATCH -p C064M0256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9
#SBATCH --time=100:00:00

set -euo pipefail

source ~/.bashrc
conda activate myconda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif command -v git >/dev/null 2>&1 && git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-data/DrivAerML/data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/HiFi3D/DrivAerML_20000}"
TARGET_VERTICES="${TARGET_VERTICES:-20000}"
SOURCE_CP_NAME="${SOURCE_CP_NAME:-CpMeanTrim}"
LOG_DIR="${LOG_DIR:-logs/drivaerml_decimate}"
PY_SCRIPT="${PY_SCRIPT:-scripts/drivaerml/decimate.py}"
RANGE_SIZE=50
START_INDEX=$((SLURM_ARRAY_TASK_ID * RANGE_SIZE + 1))
END_INDEX=$((START_INDEX + RANGE_SIZE - 1))
RANGE_START_EPOCH="$(date +%s)"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

RANGE_LABEL="$(printf "%03d_%03d" "${START_INDEX}" "${END_INDEX}")"
RUN_LOG="${LOG_DIR}/range_${RANGE_LABEL}.log"
DETAIL_DIR="${LOG_DIR}/range_${RANGE_LABEL}_details"
mkdir -p "${DETAIL_DIR}"

{
    echo "# drivaerml decimate"
    echo "# started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# repo_root: ${REPO_ROOT}"
    echo "# workdir: $(pwd)"
    echo "# python: ${PYTHON_BIN}"
    echo "# script: ${PY_SCRIPT}"
    echo "# input_dir: ${INPUT_DIR}"
    echo "# output_dir: ${OUTPUT_DIR}"
    echo "# target_vertices: ${TARGET_VERTICES}"
    echo "# source_cp_name: ${SOURCE_CP_NAME}"
    echo "# range: ${START_INDEX}-${END_INDEX}"
    echo "# detail_dir: ${DETAIL_DIR}"
    echo "start_time	end_time	duration_seconds	index	status	exit_code	input	output	detail_log	message"
} > "${RUN_LOG}"

append_log() {
    local start_time="$1"
    local end_time="$2"
    local duration_seconds="$3"
    local index="$4"
    local status="$5"
    local exit_code="$6"
    local input_path="$7"
    local output_path="$8"
    local detail_log="$9"
    local message="${10}"
    printf "%s\t%s\t%s\t%03d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${start_time}" \
        "${end_time}" \
        "${duration_seconds}" \
        "${index}" \
        "${status}" \
        "${exit_code}" \
        "${input_path}" \
        "${output_path}" \
        "${detail_log}" \
        "${message}" >> "${RUN_LOG}"
}

for index in $(seq "${START_INDEX}" "${END_INDEX}"); do
    input_path="${INPUT_DIR}/boundary_${index}.vtp"
    output_path="${OUTPUT_DIR}/boundary_${index}_20k.vtp"
    detail_log="${DETAIL_DIR}/boundary_$(printf "%03d" "${index}").log"
    start_time="$(date '+%Y-%m-%d %H:%M:%S')"
    start_epoch="$(date +%s)"

    if [[ ! -f "${input_path}" ]]; then
        echo "[${index}] missing input, skipping"
        end_time="$(date '+%Y-%m-%d %H:%M:%S')"
        end_epoch="$(date +%s)"
        duration_seconds="$((end_epoch - start_epoch))"
        printf "[%s] SKIP missing input: %s\n" "${end_time}" "${input_path}" > "${detail_log}"
        append_log "${start_time}" "${end_time}" "${duration_seconds}" "${index}" "SKIP" "0" "${input_path}" "${output_path}" "${detail_log}" "missing input"
        continue
    fi

    echo "[${index}] ${input_path} -> ${output_path}"
    set +e
    {
        echo "# start: ${start_time}"
        echo "# input: ${input_path}"
        echo "# output: ${output_path}"
        echo "# target_vertices: ${TARGET_VERTICES}"
        echo "# source_cp_name: ${SOURCE_CP_NAME}"
        "${PYTHON_BIN}" "${PY_SCRIPT}" \
            "${input_path}" \
            "${output_path}" \
            "${TARGET_VERTICES}" \
            "${SOURCE_CP_NAME}"
    } > "${detail_log}" 2>&1
    exit_code=$?
    set -e
    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    end_epoch="$(date +%s)"
    duration_seconds="$((end_epoch - start_epoch))"
    summary="$(tail -n 20 "${detail_log}" | tr '\n' '|' | sed 's/|$//')"

    if [[ ${exit_code} -eq 0 ]]; then
        append_log "${start_time}" "${end_time}" "${duration_seconds}" "${index}" "DONE" "${exit_code}" "${input_path}" "${output_path}" "${detail_log}" "${summary}"
    else
        echo "[${index}] decimation failed"
        append_log "${start_time}" "${end_time}" "${duration_seconds}" "${index}" "FAIL" "${exit_code}" "${input_path}" "${output_path}" "${detail_log}" "${summary}"
    fi
done

RANGE_END_EPOCH="$(date +%s)"
RANGE_END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
{
    echo "# finished: ${RANGE_END_TIME}"
    echo "# elapsed_seconds: $((RANGE_END_EPOCH - RANGE_START_EPOCH))"
} >> "${RUN_LOG}"

echo "Finished range ${START_INDEX}-${END_INDEX}. Log: ${RUN_LOG}"
