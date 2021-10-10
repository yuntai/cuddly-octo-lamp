set -euxo pipefail

PASSWORD=qezh34quptyojlmnq2hxutmnop93zwxeseriqbaser
OPENBLAS_CORETYPE=nehalem jupyter notebook \
--NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 \
--NotebookApp.port_retries=0 --no-browser --ip=0.0.0.0 \
--NotebookApp.token=$PASSWORD
