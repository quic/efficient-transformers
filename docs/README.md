# Docs

This directory contains the instructions for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).


## Build the docs
Install the packages required for building documentation:

```sh
 pip install -r docs/requirements.txt
```

And then, change directory to docs folder to build the docs.

```sh
cd docs/
sphinx-build -M html . build
```
## Preview the docs locally
 
```bash
cd build/html
python -m http.server
```
You can visit the page with your web browser with url `http://localhost:8080`.