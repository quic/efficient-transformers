# QEfficient Nightly Pipeline - Optimization & Configuration Guide

## Overview
The Jenkinsfile has been optimized for professional standards with improved structure, error handling, and scheduled execution capabilities.

## Key Optimizations

### 1. **Scheduled Triggers** ✓
The pipeline now automatically triggers at **2:00 AM UTC** daily:
```groovy
triggers {
    cron('0 2 * * *')  // Daily at 2:00 AM UTC
}
```

### 2. **Code Quality Improvements**
- **Environment Variables**: Centralized variable definitions to reduce duplication
- **Consistent Formatting**: Improved readability with proper indentation and structure
- **Descriptions**: Each stage has descriptive text for job dashboard
- **Error Handling**: Better error handling with graceful fallback mechanisms
- **Logging**: Enhanced logging with timestamps and status indicators

### 3. **Structural Improvements**
- **Single Preparation Stage**: Consolidated container setup and dependency installation
- **Organized Test Stages**: Logically grouped test categories
- **Better Cleanup**: Selective workspace cleanup using patterns
- **Professional Post-Build**: Status-specific messages (success, failure, unstable)

### 4. **Performance Optimizations**
- **Concurrent Test Execution**: Added parallel test argument (`-n 4`)
- **Selective Cleanup**: Only deletes test artifacts, preserves `.git`
- **Build Discard Policy**: Keeps last 30 builds or 15 days of history
- **Single Day Timeout**: Global timeout of 1 day to prevent hanging builds

## Configuration Guide

### Changing the Schedule

Edit the `triggers` section to set a custom schedule. The cron format is:
```
(minute) (hour) (day_of_month) (month) (day_of_week)
```

**Common Examples:**
```groovy
// Every day at 3:00 AM UTC
cron('0 3 * * *')

// Every Monday-Friday at 2:00 AM UTC
cron('0 2 * * 1-5')

// Every 6 hours
cron('0 */6 * * *')

// Twice daily: 2:00 AM and 2:00 PM UTC
cron('0 2,14 * * *')

// Every Sunday at midnight UTC
cron('0 0 * * 0')
```

### Additional Trigger Options

To also trigger on repository push, uncomment in the triggers section:
```groovy
triggers {
    cron('0 2 * * *')
    githubPush()  // Uncomment this line
}
```

### Environment Variables Configuration

Key variables can be customized in the `environment` section:
```groovy
environment {
    DOCKER_IMAGE = "${DOCKER_LATEST}:master_latest"      // Docker image version
    VENV_PATH = 'preflight_qeff'                          // Virtual environment path
    TOKENIZERS_PARALLELISM = 'false'                      // Tokenizer parallelism
    HF_HUB_CACHE = '/huggingface_hub'                    // HuggingFace cache location
    PYTEST_ARGS = '--durations=10 -n 4'                 // Pytest arguments
    DOCKER_USER = 'ubuntu'                                // Docker user for cleanup
}
```

## Pipeline Stages

### 1. **Prepare Environment** (40 min)
- Launches Docker container with GPU/QAIC support
- Installs Python 3.10 virtual environment
- Installs core QEfficient library and dependencies
- Installs audio, vision, and ML packages

### 2. **Unit & Integration Tests** (Parallel: 40-120 min)
- **Model Export & ONNX Tests**: Core model conversion testing
- **QAIC LLM Tests**: Language model optimization tests
- **QAIC Feature Tests**: QAIC-specific feature validation

### 3. **QAIC MultiModal Tests** (120 min)
- Vision-language model testing
- Image-to-text inference validation

### 4. **QAIC Diffusion Models Tests** (120 min)
- Diffusion model compilation and optimization
- Generative model testing

### 5. **CLI Inference Tests** (120 min)
- Command-line interface testing
- End-to-end inference validation

### 6. **Finetune CLI Tests** (20 min)
- Fine-tuning capability validation
- QAIC PyTorch integration testing

## Pipeline Features

### Health Checks & Reporting
✓ **JUnit Test Results**: Automatic collection and reporting  
✓ **Build History**: Last 30 builds retained  
✓ **Timestamped Logs**: All logs include timestamps  
✓ **Concurrent Build Prevention**: Avoids duplicate test execution  
✓ **Selective Cleanup**: Smart workspace cleanup  

### Status Indicators
- ✓ Success: All tests passed
- ✗ Failure: One or more stages failed
- ⚠ Unstable: Tests ran but some were skipped

## Monitoring

### Jenkins Dashboard
1. Navigate to the pipeline job in Jenkins
2. Check "Build History" for recent executions
3. Click on a build to view detailed logs
4. Check "Console Output" for real-time progress

### Test Results
- Test reports are automatically parsed from `tests/tests_log.xml`
- Results appear in the "Test Result" section of each build
- Failed tests are highlighted for quick debugging

## Troubleshooting

### Pipeline Fails to Trigger
- Verify cron syntax is correct
- Check Jenkins system time matches UTC
- Ensure "Poll SCM" is not conflicting with cron trigger

### Test Timeouts
- Increase timeout in specific stage: `timeout(time: XX, unit: 'MINUTES')`
- Or globally in options: `timeout(time: 2, unit: 'DAYS')`

### Docker Container Issues
- Check Docker daemon is running: `sudo service docker status`
- Verify node label matches: `qeff_node`
- Check disk space: build artifacts can be large

### Permission Errors
- Verify Ubuntu user exists on agent
- Check Docker socket permissions
- Ensure Jenkins user can run sudo commands

## Best Practices

1. **Monitor Regularly**: Check pipeline status at least weekly
2. **Archive Logs**: Keep build logs for audit trail
3. **Update Dependencies**: Review and update pip packages quarterly
4. **Scale Tests**: Add more parallel workers for faster execution
5. **Backup Tests**: Keep a copy of critical test files

## Future Enhancements

Consider implementing:
- [ ] Email notifications on build failure
- [ ] Performance metrics collection
- [ ] Test report artifacts archiving
- [ ] Slack integration for status updates
- [ ] Coverage report generation
- [ ] Performance regression detection
