ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace
ADD run_colab.sh /colab/run_colab.sh
WORKDIR /workspace

RUN pip install --upgrade "jupyter_http_over_ws>=0.0.7" && jupyter serverextension enable --py jupyter_http_over_ws
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension
RUN pip install -r requirements.txt

EXPOSE 8888
ENTRYPOINT ["/bin/bash", "/colab/run_colab.sh"]
