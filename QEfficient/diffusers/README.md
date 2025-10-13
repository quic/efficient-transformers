
<div align="center">


# **Diffusion Models on Qualcomm Cloud AI 100**


<div align="center">

### 🎨 **Experience the Future of AI Image Generation**

* Optimized for Qualcomm Cloud AI 100*

<img src="../../docs/image/girl_laughing.png" alt="Sample Output" width="400">

**Generated with**: `stabilityai/stable-diffusion-3.5-large` • `"A girl laughing"` • 28 steps • 2.0 guidance scale •  ⚡



</div>



[![Diffusers](https://img.shields.io/badge/Diffusers-0.31.0-orange.svg)](https://github.com/huggingface/diffusers)
</div>

---

## ✨ Overview

QEfficient Diffusers brings the power of state-of-the-art diffusion models to Qualcomm Cloud AI 100 hardware for text-to-image generation. Built on top of the popular HuggingFace Diffusers library, our optimized pipeline provides seamless inference on Qualcomm Cloud AI 100 hardware.

## 🛠️ Installation

### Prerequisites

Ensure you have Python 3.8+ and the required dependencies:

```bash
# Create Python virtual environment (Recommended Python 3.10)
sudo apt install python3.10-venv
python3.10 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip
```

### Install QEfficient

```bash
# Install from GitHub (includes diffusers support)
pip install git+https://github.com/quic/efficient-transformers

# Or build from source
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/qefficient-0.0.1.dev0-py3-none-any.whl
```

### Install Diffusers Dependencies

```bash
# Install diffusers optional dependencies
pip install "QEfficient[diffusers]"
```

---

## 🎯 Supported Models

### Stable Diffusion 3.x Series
- ✅ [`stabilityai/stable-diffusion-3.5-large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- ✅ [`stabilityai/stable-diffusion-3.5-large-turbo`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
---


## 📚 Examples

Check out our comprehensive examples in the [`examples/diffusers/`](../../examples/diffusers/) directory:

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
pip install -e ".[diffusers,test]"
```

---

## 🙏 Acknowledgments

- **HuggingFace Diffusers**: For the excellent foundation library
- **Stability AI**: For the amazing Stable Diffusion models  
---

## 📞 Support

- 📖 **Documentation**: [https://quic.github.io/efficient-transformers/](https://quic.github.io/efficient-transformers/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/quic/efficient-transformers/issues)

---

