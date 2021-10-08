FROM python:3
WORKDIR /root
COPY . /root/
RUN pip3 install virtex fasttext gunicorn requests
EXPOSE 580/tcp
ENTRYPOINT ["/bin/bash", "-l", "query_matching_demo.sh"]