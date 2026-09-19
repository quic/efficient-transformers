#!/usr/bin/env bash

set -euo pipefail

output='tests/tests_log.xml'
temporary_output="${output}.tmp"
sources=()

for source in tests/tests_log*.xml; do
    [[ -f "${source}" ]] || continue
    case "${source}" in
        "${output}"|tests/tests_log_disagg_batch_*.xml)
            continue
            ;;
    esac
    sources+=("${source}")
done

if ((${#sources[@]} == 0)); then
    printf '%s\n' '<?xml version="1.0" encoding="utf-8"?><testsuites></testsuites>' > "${output}"
    exit 0
fi

rm -f "${temporary_output}"
junitparser merge "${sources[@]}" "${temporary_output}"
mv "${temporary_output}" "${output}"
