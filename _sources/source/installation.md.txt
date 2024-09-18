# Pre-requisites
System Requirements:
1. [Supported Linux OS](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/#operating-systems) - Ubuntu, RHEL and AWS Linux
2. [Cloud AI 100 Platform and Apps SDK installed](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/) 
3. [SDK Pre-requisites](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Pre-requisites/pre-requisites/) 
4. [Multi-device support enabled for model sharding](https://github.com/quic/cloud-ai-sdk/tree/1.12/utils/multi-device)

# Linux Installation 
There are two different way to install efficient-transformers.

## Using SDK

* Download Apps SDK: [Cloud AI 100 Platform and Apps SDK install](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/)  


```bash
# Install using Apps SDK

bash install.sh --enable-qeff
source  /opt/qti-aic/dev/python/qeff/bin/activate

```
## Using GitHub Repository

```bash
# Create Python virtual env and activate it. (Required Python 3.8)

python3.8 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient Repo.
pip install git+https://github.com/quic/efficient-transformers --extra-index-url https://download.pytorch.org/whl/cpu

# Or build wheel package using the below command.
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/QEfficient-0.0.1.dev0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cpu

``` 

# Sanity Check

After any of the above installation methods, you can check if ``QEfficient`` is installed correctly by using
```bash
python -c "import QEfficient; print(QEfficient.__version__)"
```
If the above line executes successfully, you are good to go ahead and start deploying models on ``Cloud AI 100`` cards using ``QEfficient`` library.
