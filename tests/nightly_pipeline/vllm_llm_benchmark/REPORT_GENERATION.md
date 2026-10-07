# vLLM QAIC Benchmark Report Generation and Email Distribution

This document describes the complete workflow for generating and distributing benchmark reports.

## Overview

The vLLM QAIC benchmark pipeline now includes automated report generation and email distribution:

1. **Benchmark Execution**: Run all benchmark categories (LLM, embedding, audio, VLM)
2. **CSV Consolidation**: Merge all result CSVs into a single consolidated published CSV
3. **HTML Report Generation**: Create a professional HTML report from the consolidated CSV
4. **Email Distribution**: Send the report to specified email recipients

## Workflow Steps

### Step 1: Run Benchmarks

Execute the benchmark pipeline with your desired configurations:

```bash
# Via Jenkins UI or CLI
# Set parameters:
# - RUN_DEFAULT=true, RUN_EMBEDDING=true, RUN_AUDIO=true, RUN_VLM=true
# - EMAIL_RECIPIENTS=team@example.com (comma-separated for multiple recipients)
```

### Step 2: Consolidate Results (Automatic)

After all benchmark stages complete, the "Generate Report" stage automatically:

1. **Merges all result CSVs** using `merge_published_results.py`:
   - Reads all `*_results.csv` files from the results directory
   - Consolidates into a single `consolidated_published_results.csv`
   - Includes 17 key fields: model, model_category, config_name, config_summary, status, mean_ttft_s, mean_tpot_s, mean_itl_s, decode_TPS, request_throughput_req_s, vllm_qaic_branch, qaic_disagg_branch, qserve_branch, qeff_branch, qaic_sdk_version, server_command, client_command

2. **Generates HTML report** using `generate_html_report.py`:
   - Creates a professional HTML report with styling
   - Includes environment information section (branch details, SDK version)
   - Includes test results summary (total, passed, failed counts)
   - Includes detailed test results table with all metrics

### Step 3: Email Distribution (Automatic)

If `EMAIL_RECIPIENTS` parameter is set:

- **On Success**: Sends HTML report with both HTML and CSV attachments
- **On Failure**: Sends failure notification with build details link

## Jenkins Configuration

### Parameters

Add the following parameter to your Jenkins job:

```groovy
string(name: 'EMAIL_RECIPIENTS', defaultValue: '', description: 'Email recipients for benchmark report (comma-separated). Empty = no email sent.')
```

### Usage

1. Open the Jenkins job configuration
2. Set `EMAIL_RECIPIENTS` to your email group (e.g., `team@example.com` or `user1@example.com,user2@example.com`)
3. Run the build
4. After all benchmarks complete, the report will be automatically generated and emailed

## Local Testing

### Generate Consolidated CSV

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/merge_published_results.py \
  --results-dir /path/to/results \
  --output /path/to/consolidated_published_results.csv
```

### Generate HTML Report

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/generate_html_report.py \
  --csv /path/to/consolidated_published_results.csv \
  --output /path/to/benchmark_report.html
```

## Report Contents

### Environment Information Section

Displays:
- vLLM QAIC Branch
- QAIC Disagg Branch
- QServe Branch
- QEff Branch
- QAIC SDK Version

### Test Results Summary

Shows:
- Total number of tests
- Number of passed tests
- Number of failed tests

### Detailed Test Results Table

Columns:
- Model Name
- Category (LLM, Embedding, Audio, VLM)
- Config (config_name)
- Config Summary
- Status (✓ PASS / ✗ FAIL)
- Mean TTFT (s)
- Mean TPOT (s)
- Mean ITL (s)
- Decode TPS
- Request Throughput (req/s)

## Files

- `merge_published_results.py` - Consolidates all result CSVs
- `generate_html_report.py` - Generates HTML report from consolidated CSV
- `JenkinsfileVllmLlmBenchmark` - Updated with report generation and email stages

## Email Template

The email includes:
- Subject: `vLLM QAIC Benchmark Report - Build #<BUILD_NUMBER>`
- Body: Full HTML report rendered directly in the email (not as attachment)
- Attachments: 
  - `consolidated_published_results.csv` - Consolidated results data for reference

## Troubleshooting

### Report not generated

1. Check that all benchmark stages completed successfully
2. Verify the results directory contains `*_results.csv` files
3. Check Jenkins logs for the "Generate Report" stage

### Email not sent

1. Verify `EMAIL_RECIPIENTS` parameter is set (not empty)
2. Check Jenkins email configuration (Manage Jenkins > Configure System > Email Notification)
3. Verify the email plugin is installed and configured
4. Check Jenkins logs for email delivery errors

### Missing data in report

1. Verify all benchmark categories ran successfully
2. Check that result CSVs contain data (not empty)
3. Verify environment variables are properly set in result CSVs
