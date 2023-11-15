ARG PYTHON_VERSION=3.11.6
ARG DEBIAN_VERSION=bullseye

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION}
ARG PYTHON_VERSION
ARG DEBIAN_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY requirements.txt /tmp/src/requirements.txt
RUN pip install --no-cache-dir -r /tmp/src/requirements.txt

COPY . /tmp/src
RUN pip install --no-cache-dir /tmp/src && rm -rf /tmp/src

ENTRYPOINT ["python"]
