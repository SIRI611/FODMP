ORG = huawei

# Name of image
IMAGE = pineapple

# Docker TAG
TAG = latest

all: pre_build build after_build

gui: pre_build build after_build run_gui

headless: pre_build build after_build run_headless

pre_build:
	cp -r ../app/ .

build:
	docker build \
		-t $(ORG)/$(IMAGE):$(TAG) \
		--network=host \
		--build-arg http_proxy=http://localhost:3128 \
		--build-arg https_proxy=https://localhost:3128 \
		--build-arg no_proxy \
		.

after_build:
	rm -r app/

run_gui:
	./run_gui_container.sh

run_headless:
	./run_headless_container.sh

clean_gui:
	docker stop gui_pineapple
	docker rm gui_pineapple
	docker rmi huawei/pineapple:latest
	docker images
	docker ps -a

clean_headless:
	docker stop headless_pineapple
	docker rm headless_pineapple
	docker rmi huawei/pineapple:latest
	docker images
	docker ps -a
