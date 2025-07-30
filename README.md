# Sustainable Geo-Distributed AI Inference


### Installing Highs solver
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
export PATH="$PATH:$(pwd)"
./vcpkg install highs
```

### Running STOP
Clone the repository and then run the following commands:
```
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .
cd ..
./build/STOP --method STOP --model_name Yolo --n_machines 3 --max_demands_per_round 1000 --seed 0 --C_avg 0.001 --R_avg 2 --V 500
```