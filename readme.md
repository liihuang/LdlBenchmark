# Ldl Benchmark

## Description
This is a benchmark for demonstrating the efficiency of our optimized LDL solver as part of

[**Towards Realtime: A Hybrid Physics-based Method for Hair Animation on GPU**](https://lihuang.work/realtime_hair)(*Li Huang, Fan Yang, Chendi Wei, Yuju (Edwin) Chen, Chun Yuan, Ming Gao*).

## Usage
```bash
mkdir build
cd build
cmake ..
make
```

## Performance
The result of the benchmark may vary with different platforms. We have tested it on two platforms, the results (total time cost of each solver solving 30000 linear systems) are list below:
|            | CPU(Eigen) | Naive GPU | Optimized GPU |
|:----------:|:----------:|:---------:|:-------------:|
| Platform 1 | 1723.19 ms |  71.06 ms |    8.75 ms    |
| Platform 2 |  910.28 ms |  66.91 ms |    6.09 ms    |


(*Platform 1: Intel i9-9900K + Nvidia RTX 2080Ti*, *Platform 2: Intel i5-13600KF + Nvidia RTX 4070*)
## BibTex
Comming soon.