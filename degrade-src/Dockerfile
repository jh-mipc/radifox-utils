FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends git python3 python3-pip

COPY . /tmp/src

RUN pip install /tmp/src && rm -rf /tmp/src

ENTRYPOINT ["apply-degrade"]
