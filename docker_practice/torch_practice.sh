nvidia-docker run -it -h torch_practice \
	-p 1255:1255 \
	--ipc=host \
	--name torch_practice \
	-v ~/Projects:/projects \
	-v ~/datasets:/datasets \
	tootouch/torch_practice bash
