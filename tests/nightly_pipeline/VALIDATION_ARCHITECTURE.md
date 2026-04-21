# Validation System Architecture

## 📊 Validation Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE EXECUTION START                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   PHASE 1    │  │   PHASE 2    │  │   PHASE 3    │
            │   EXPORT     │  │  COMPILE     │  │  INFERENCE   │
            │  (Parallel)  │  │  (Parallel)  │  │ (Sequential) │
            └──────────────┘  └──────────────┘  └──────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   SAVE RESULTS TO JSON       │
                        │  (pipeline_results.json)     │
                        └──────────────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │    START VALIDATION                  │
                   │  (if baseline_file provided)         │
                   └──────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
    ┌─────────────┐          ┌──────────────────┐         ┌──────────────┐
    │Load Baseline│          │  For Each Model  │         │ No Baseline? │
    │   Results   │          │                  │         │              │
    └─────────────┘          │ 1. Compare Perf  │         │SKIP & PASS✓  │
         │                   │    Metrics       │         └──────────────┘
         │                   │    (±5%)         │
         │                   │ 2. Compare Gen   │
         │                   │    IDs (≥0.95)   │
         │                   └──────────────────┘
         │                            │
         └────────────────────────────┤
                                      ▼
                    ┌──────────────────────────────┐
                    │  COLLECT VALIDATION RESULTS  │
                    │  (per-model pass/fail)       │
                    └──────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
          ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
          │  ALL PASS? ✓     │ │  SOME FAIL?  │ │  SKIPPED?    │
          │                  │ │  ✗           │ │  ⊘           │
          │ YES              │ │ YES          │ │ YES          │
          └──────────────────┘ └──────────────┘ └──────────────┘
                    │                 │                 │
                    │                 │                 │
                    ▼                 ▼                 ▼
          ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
          │  UPDATE          │ │  KEEP        │ │  PASS WITH   │
          │  BASELINE        │ │  BASELINE    │ │  NO CHANGES  │
          │  (Replace File)  │ │  (Unchanged) │ │  (No file)   │
          └──────────────────┘ └──────────────┘ └──────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │  LOG VALIDATION REPORT       │
                        │  (Detailed Results)          │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │  RETURN STATUS               │
                        │  (True = Pass or Skip)       │
                        │  (False = Fail)              │
                        └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │  PIPELINE EXECUTION END      │
                        └──────────────────────────────┘
```

---

## 📂 Data Flow for Validation

```
┌────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION INPUT                         │
│  (configs/causal_pipeline_config.json)                        │
├────────────────────────────────────────────────────────────────┤
│  {                                                             │
│    "config_name": "standard_config",                           │
│    "baseline_file": "configs/baseline_results.json"   ◄────────┼──┐
│    "compile_params": {...}                                    │  │
│    "generation_params": {...}                                 │  │
│  }                                                             │  │
└────────────────────────────────────────────────────────────────┘  │
                                                                     │
┌──────────────────────────────────────────────────────────────────┐│
│                      BASELINE FILE                               ││
│  (configs/baseline_inference_results.json)                       ││
├──────────────────────────────────────────────────────────────────┤│
│  {                                                               ││
│    "inferences": [                                              ││
│      {                                                          ││
│        "model_name": "gpt2",                                    ││
│        "prefill_time": 0.15,         ◄─┐                       ││
│        "decode_perf": 85.5,          ◄─┤ Compared              ││
│        "total_perf": 120.3,          ◄─┤ with ±5%             ││
│        "total_time": 1.25,           ◄─┤ tolerance             ││
│        "generated_ids": [...]        ◄─┤ Compared              ││
│      }                               ◄─┤ with cosine           ││
│    ]                                 ◄─┤ similarity            ││
│  }                                   ◄─┘                       ││
└──────────────────────────────────────────────────────────────────┘│
                                                                     │
┌──────────────────────────────────────────────────────────────────┐│
│              PIPELINE EXECUTION & CURRENT RESULTS                ││
│  (nightly_pipeline_outputs/results/pipeline_results.json)        ││
├──────────────────────────────────────────────────────────────────┤│
│  {                                                               ││
│    "timestamp": "2026-04-21 12:00:00",                          ││
│    "inferences": [                                              ││
│      {                                                          ││
│        "model_name": "gpt2",                                    ││
│        "prefill_time": 0.148,        ─────► COMPARE ─┐         ││
│        "decode_perf": 84.2,          ─────► WITH     │ ┌─►     ││
│        "total_perf": 118.5,          ─────► BASELINE │ │       ││
│        "total_time": 1.235,          ─────► VALUES   │ │       ││
│        "generated_ids": [...]        ─────► TOLERANCE│ │       ││
│      }                               ─────► CHECK    │ │       ││
│    ]                                       └─────────┼─┘       ││
│  }                                                  │           ││
└──────────────────────────────────────────────────────────────────┘│
                                                      │              │
                                                      ▼              │
                                          ┌───────────────────────┐ │
                                          │ VALIDATION RESULTS    │ │
                                          ├───────────────────────┤ │
                                          │ Model: gpt2           │ │
                                          │ Status: passed        │ │
                                          │                       │ │
                                          │ Metrics:              │ │
                                          │ prefill_time: 1.33%   │ │
                                          │ decode_perf: 1.52%    │ │
                                          │ total_perf: 1.50%     │ │
                                          │ total_time: 1.20%     │ │
                                          │                       │ │
                                          │ Gen IDs Similarity:   │ │
                                          │ 0.9876 (✓ >= 0.95)    │ │
                                          └───────────────────────┘ │
                                                      │              │
                                                      ▼              │
                                          ┌───────────────────────┐ │
                                          │ ALL PASS?             │ │
                                          └───────────────────────┘ │
                                                      │              │
                                        ┌─────────────┴─────────────┤
                                        │ YES: Update Baseline ──┘
                                        │ (Replace with current)
                                        │
                                        └──► Baseline File Updated
                                             for next run
```

---

## 🔄 State Transitions

```
                    ┌──────────────────────────┐
                    │  BASELINE NOT FOUND      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  SKIP VALIDATION         │
                    │  Return: PASS (True)     │
                    └─────────────────────────-┘
                                 
     
     ┌──────────────────────────────────────────────────────┐
     │         BASELINE FOUND - START VALIDATION            │
     └────────────────────┬─────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     ┌──────────────────┐    ┌──────────────────┐
     │  ALL PASS        │    │  ANY FAIL        │
     │  (±5%, ≥0.95)    │    │  (>5%, <0.95)    │
     └────────┬─────────┘    └────────┬─────────┘
              │                       │
              ▼                       ▼
     ┌──────────────────┐    ┌──────────────────┐
     │ UPDATE BASELINE  │    │ KEEP BASELINE    │
     │ Return: PASS ✓   │    │ Return: FAIL ✗   │
     └──────────────────┘    └──────────────────┘
```

---

## 📊 Tolerance Boundaries

```
Performance Metric Example: prefill_time

   0.15s (Baseline)
    │
    ├─ Acceptable Range (±5%)
    │  ├─ 5% Lower Bound: 0.1425s
    │  │
    │  ├─ 0.14s ✓ PASS
    │  ├─ 0.145s ✓ PASS
    │  ├─ 0.15s ✓ PASS (exact match)
    │  ├─ 0.155s ✓ PASS
    │  ├─ 0.1575s ✓ PASS
    │  │
    │  └─ 5% Upper Bound: 0.1575s
    │
    └─ Unacceptable Range (> 5%)
       ├─ 0.14s ✗ FAIL (6.7% below)
       ├─ 0.158s ✗ FAIL (5.3% above)
       ├─ 0.13s ✗ FAIL (13.3% below)
       └─ 0.20s ✗ FAIL (33.3% above)


Generated IDs Similarity Example

   Cosine Similarity Range: 0 to 1

   0.95 (Threshold)
    │
    ├─ Acceptable Range (≥ 0.95)
    │  ├─ 0.95 ✓ PASS (exact threshold)
    │  ├─ 0.97 ✓ PASS
    │  ├─ 0.99 ✓ PASS (near perfect)
    │  └─ 1.00 ✓ PASS (perfect match)
    │
    └─ Unacceptable Range (< 0.95)
       ├─ 0.94 ✗ FAIL (1% below)
       ├─ 0.90 ✗ FAIL (5% below)
       ├─ 0.80 ✗ FAIL (15% below)
       └─ 0.50 ✗ FAIL (50% different)
```

---

## 📋 Class Relationships

```
┌─────────────────────────────────────────────────────────┐
│           NightlyPipeline (Main Pipeline)               │
├─────────────────────────────────────────────────────────┤
│ - run_nightly_pipeline_configs.py                       │
│ - nightly_causal_lm_pipeline.py                         │
│                                                         │
│ Methods:                                                │
│ • phase_1_export_parallel()                             │
│ • phase_2_compile_parallel()                            │
│ • phase_3_inference_sequential()                        │
│ • save_results() → returns results dict                 │
│ • run() → calls validator, returns bool                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ uses
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         ResultValidator (Validation Engine)             │
├─────────────────────────────────────────────────────────┤
│ - result_validator.py                                   │
│                                                         │
│ Attributes:                                             │
│ • baseline_file: Optional[str]                          │
│ • baseline_results: Dict                                │
│ • TOLERANCE_PERCENT: float = 5.0                        │
│                                                         │
│ Methods:                                                │
│ • validate_results(inferences)                          │
│   → List[ValidationResult], bool                        │
│ • _validate_perf_metrics(current, baseline, model)      │
│ • _validate_generated_ids(current, baseline, model)     │
│ • _cosine_similarity(a, b) → float                      │
│ • update_baseline(results, output_file)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ produces
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│        ValidationResult (Validation Outcome)            │
├─────────────────────────────────────────────────────────┤
│ Dataclass:                                              │
│ • model_name: str                                       │
│ • status: str  ("passed", "failed", "skipped")          │
│ • perf_metrics_summary: Dict                            │
│ • generated_ids_summary: Dict                           │
│ • error_message: Optional[str]                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Structure

```
Pipeline Config File
└── pipeline_configs[]: List
    └── [0]: Config Dictionary
        ├── config_name: "standard_config"
        ├── output_dir: "./nightly_pipeline_outputs"
        ├── baseline_file: "./configs/baseline_inference_results.json"  ◄─ VALIDATION
        ├── num_export_workers: 2
        ├── num_compile_workers: 4
        ├── export_params: {}
        ├── compile_params: {
        │   ├── prefill_seq_len: 32
        │   ├── ctx_len: 128
        │   ├── num_cores: 16
        │   └── aic_hw_version: "ai100"
        │}
        └── generation_params: {
            ├── generation_len: 100
            └── prompt: "My name is"
        }

Baseline Results File
└── inferences[]: List
    └── [0]: Inference Result
        ├── model_name: "gpt2"
        ├── prefill_time: 0.15  ◄─ COMPARED
        ├── decode_perf: 85.5   ◄─ WITH ±5%
        ├── total_perf: 120.3   ◄─ TOLERANCE
        ├── total_time: 1.25    ◄─ TOLERANCE
        └── generated_ids: [...] ◄─ COSINE SIMILARITY
```

---

## 💡 Key Concepts

```
Deviation Percentage:
  How much current value differs from baseline
  Formula: |current - baseline| / baseline × 100%
  Tolerance: ±5%

Cosine Similarity:
  How similar two sequences are (0 = different, 1 = identical)
  Formula: (A · B) / (||A|| × ||B||)
  Threshold: ≥ 0.95

Mean Absolute Deviation (MAD):
  Average absolute difference in token IDs
  Used to understand token-level variations

Status Indicators:
  ✓ = Passed all validations
  ✗ = Failed one or more validations
  ⊘ = Skipped (no baseline to compare)
```
