FROM nvidia/cuda:10.2-base

RUN apt-get update 
RUN apt-get install -y python3
RUN apt-get install -y python3-pip && pip3 install --upgrade pip
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y git
RUN apt-get install -y nano

ADD ./requirements.txt /home/app/install/requirements.txt

RUN cd /home/app/install/ &&  pip3 install -r requirements.txt

RUN pip3 install torch torchvision
RUN pip3 install tensorflow

