#!/bin/bash
#!/bin/bash

sudo docker run -d --rm -it --ipc=host --gpus all -v $(pwd):/usr/src/wire-art -p 0.0.0.0:5005:1122 --name apes apes:v1.6
sudo docker exec -it apes bash
# sudo docker run -d -it --ipc=host --gpus all -v $(pwd)/..:/usr/src/wire-art -p 0.0.0.0:7007:6006 --name wa-c wire-art:2.0
#  forward two ports 22 (for ssh) and 6006 (for Tensorboard)