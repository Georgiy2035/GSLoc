# Installation

## Quick experiments (without OpenPlaceRecognition)

For experiments that don't require `OpenPlaceRecognition`, it can be much simpler and faster to set up a small `venv` with a minimal set of dependencies.
For example, this is convenient for EDA notebooks and various visualizations.

For such purposes, the `uv` utility is used - https://github.com/astral-sh/uv

The project uses a `uv.lock` file that contains all exact dependency versions. Simply run:

```bash
uv sync
```

The environment is automatically used when you run commands with `uv run`, but you can also activate it manually:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or use uv run for individual commands
uv run python your_script.py
uv run jupyter notebook
```

## Docker environment for OpenPlaceRecognition

For all experiments using the [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) library,
a docker image is created for this library:

### Building the base image

0. Check that you have downloaded the submodules

    ```bash
    git submodule update --init --recursive
    ```

1. Either download from dockerhub:

    ```bash
    docker pull alexmelekhin/open-place-recognition:base
    ```

    Or build the `base` image manually (not recommended, it takes a long time):

    ```bash
    bash libs/OpenPlaceRecognition/docker/build_base.sh
    ```

**Comment:** if you look at the contents of the base image, you might be horrified by the number of heavy dependencies. Many of them are used in a single secondary module of the library, and ideally all these dependencies should be made optional - the library should work without installing them. **We are striving for this**, in the near future plans - to "untie" from all unnecessary heavy dependencies. Ideally - to bring the library to the possibility of installation with a simple `pip install`.

### Building the `devel` image and running the container

The next steps are already inside this repository (so that you can quickly add dependencies needed specifically for this project to the `devel` version)

1. Build the `multimodal-place-recognition:devel` image (an extension over `alexmelekhin/open-place-recognition:base`):

    ```bash
    bash docker/build_devel.sh
    ```

2. Run the container, specifying the path to the data directory (it will be mounted as a volume with `rw` permissions):

    ```bash
    bash docker/start.sh YOUR_DATA_DIR
    ```

3. It is recommended to connect to the terminal inside the container using the `into.sh` script (look at its contents, the whole point is in the `--user docker_mmpr` flag):

    ```bash
    bash docker/into.sh
    ```

Inside the container there will be two mounted directories with `rw` permissions:

- `/home/docker_mmpr/multimodal-place-recognition` - current repository
- `/home/docker_mmpr/Datasets` - data directory

### Installing the library and third party inside the container

This is a short retelling of the library documentation - https://openplacerecognition.readthedocs.io/en/latest/#third-party-packages

```bash
pip install -e ~/multimodal-place-recognition/libs/OpenPlaceRecognition

cd ~/multimodal-place-recognition/libs/OpenPlaceRecognition/third_party/GeoTransformer
sudo bash setup.sh  # default password is `user`, you can change it by passing build-arg to `docker build`

cd ~/multimodal-place-recognition/libs/OpenPlaceRecognition/third_party/HRegNet/hregnet/PointUtils
sudo python setup.py install
cd ../..  # go back to the third_party/HRegNet directory
sudo pip install .

cd ~
```

Check that `opr` imports: `python -c "import opr; print(opr.__version__)"`

You are magnificent! 🎉
