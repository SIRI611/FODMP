#!/bin/bash

set -e

# Collect proxy environment variables to pass to docker build
function build_args_proxy() {
  # Enable host-mode networking if necessary to access proxy on localhost
  env | grep -iP '^(https?|ftp)_proxy=.*$' | grep -qP 'localhost|127\.0\.0\.1' && {
    echo -n "--network host "
  }
  # Proxy environment variables as "--build-arg https_proxy=https://..."
  env | grep -iP '^(https?|ftp|no)_proxy=.*$' | while read env_proxy; do
    echo -n "-e ${env_proxy} "
  done
}
echo $(build_args_proxy)

cd ../app

xhost +local:root

docker run \
    -it \
    -d $(build_args_proxy) \
    --env DISPLAY \
    --gpus all \
	  --name gui_pineapple \
	  --privileged \
		--volume /tmp/.X11-unix:/tmp/.X11-unix \
		--volume $(pwd):/usr/src/app \
	  huawei/pineapple:latest
