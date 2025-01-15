FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
# https://stackoverflow.com/questions/46064433/cuda-home-path-for-tensorflow
ENV CUDA_HOME=/usr/local/cuda-11.6
ENV FORCE_CUDA="1"

WORKDIR /usr/src/wire-art

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

RUN apt-get install -y vim git
RUN apt-get install -y wget

RUN apt-get install -y python3-pip
RUN apt-get install -y python3-opencv

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init
RUN source ~/.bashrc
ENV PATH /root/miniconda/bin:$PATH

RUN conda update --all
RUN conda create -n APES python=3.9 -y
RUN echo "conda activate APES" >> ~/.bashrc

RUN conda install -n APES pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# RUN conda install --name pytorch3d --yes openblas-devel -c anaconda
# RUN conda install --name pytorch3d --yes pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install --name APES --yes -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install --name APES --yes -c bottler nvidiacub  

RUN conda install --name APES --yes pytorch3d -c pytorch3d  && \
    conda install --name APES --yes jupyter  

COPY requirements.txt requirements.txt
RUN /bin/bash -c ". activate APES && \
    pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt &&\
    pip3 install --no-cache-dir scikit-learn &&\
    pip3 install open3d &&\
    pip3 install -U networkx[default]"
    #pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 &&\

RUN apt install unzip
RUN ln -s /usr/bin/python3 /usr/bin/python

EXPOSE 8000
