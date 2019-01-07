FROM python:3.6-stretch

ADD docker/debian-requirements.txt debian-requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
   DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" debian-requirements.txt) && \
   apt-get install -y --force-yes $DEBIAN_PACKAGES && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/* && \
   rm debian-requirements.txt

ADD docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && rm requirements.txt

ADD docker/docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH /workspace/:/workspace/src:$PYTHONPATH

ENTRYPOINT ["./docker-entrypoint.sh"]