#!/bin/bash

docker exec --user docker_mmpr -it ${USER}_mmpr \
    /bin/bash -c "cd /home/docker_mmpr; echo ${USER}_mmpr container; echo ; /bin/bash"
