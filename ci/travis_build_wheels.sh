#!/bin/bash
set -e -x

# Install librdrm developement headers
yum install -y libdrm-devel

# Setup python versions to be built
declare -a python_versions
python_versions=(cp35-cp35m cp36-cp36m cp37-cp37m cp38-cp38)

# Install dependencies and compile wheels
for version in "${python_versions[@]}"; do
    /opt/python/$version/bin/pip install -r /io/dev-requirements.txt
    /opt/python/$version/bin/pip wheel /io/ -w dist-wip/
done

# Bundle external shared libraries into the wheels
for whl in dist-wip/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done

# Install wheels
for version in "${python_versions[@]}"; do
    /opt/python/$version/bin/pip install pyamdgpuinfo --no-index -f /io/dist
done
