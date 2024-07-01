# Docs

This directory contains the stuff for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).


## Build the docs
Firstly, install the packages:

```sh
python3 -m pip install -r ./requirements.txt
```


And then, make the docs:

```sh
make html
```
## Preview the docs locally

The basic way to preview the docs is using the `http.server`:

```sh
cd build/html

python3 -m http.server 8080
```

And you can visit the page with your web browser with url `http://localhost:8080`.