# granite.cpp
It is a toy! granite inference is implemented in pure C++ without any dependency.

## download model

```bash
curl -L -o tools/hfd.sh https://hf-mirror.com/hfd/hfd.sh
chmod +x tools/hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./tools/hfd.sh ibm-granite/granite-3.0-3b-a800m-instruct
```
> The model will be downloaded to `./granite-3.0-3b-a800m-instruct` directory.

## build

```bash
mkdir build && cd build
cmake ..
make -j
```

## run

```bash
# generate text
./build/granite generate "Describe the iphone design" granite-3.0-3b-a800m-instruct  100 0.8 40 42

# reference output:
The iPhone design is characterized by a sleek, minimalist, and modern aesthetic. It features a glass front and back, with a stainless steel frame that gives it a premium feel. The display is a high-resolution Retina Display, which provides crisp and clear visuals. The iPhone has a slim and lightweight design, making it easy to hold and use. The device also has a fingerprint sensor, located on the home button, which
```
