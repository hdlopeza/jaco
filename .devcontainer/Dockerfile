FROM tensorflow/tensorflow:nightly-gpu-jupyter

RUN python -c "import tensorflow as tf; print(f'Tensorflow version: {tf.__version__}')"

RUN apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        less \
        zip && \
    rm -rf /var/lib/apt/lists/*

# Trick pip into thinking that the 'tensorflow' package is installed.
# Installing `object_detection` attempts to install the 'tensorflow' package.
# Name the symlink with the suffix from tf_nightly_gpu.
WORKDIR /usr/local/lib/python3.6/dist-packages
RUN ln -s tf_nightly_gpu-* tensorflow-$(ls -d1 tf_nightly_gpu* | sed 's/tf_nightly_gpu-\(.*\)/\1/')

WORKDIR /tf
RUN curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip && \
    unzip protoc-3.11.4-linux-x86_64.zip && \
    cp bin/protoc /usr/local/bin && \
    rm -r protoc-3.11.4-linux-x86_64.zip bin/

# Upgrade pip.
RUN python -m pip install --no-cache-dir --upgrade pip

RUN git clone --depth 1 https://github.com/tensorflow/models
WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install  --no-cache-dir .

RUN python -c "import tensorflow as tf; print(f'Tensorflow version: {tf.__version__}')"