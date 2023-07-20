
# Heterogeneous Graph Scheduler

This is the repository for our paper "Learning-enabled Flexible Job-Shop Scheduling for Scalable Smar Manufacturing", which is extended version of our conference paper "Graph-based Reinforcement Learning for Flexible Job Shop Scheduling with Transportation Constriaints".


# Requirements
- python 3.7.16
- gym 0.19
- pytorch 1.12.1
- dgl 1.0.2
- pyg 2.3.0
- pandas 1.3.5

Other requirements are listed in file "environment.yaml"

# Instructions

To train the HGS modules:
```
python3 train.py  --cuda 0 --log_file_desc hgs_10_6_6 --algorithm hgs --job_centric
```

To train the baseline algorithms:
```
python3 train.py  --cuda 0 --log_file_desc hgnn_10_6_6 --algorithm hgnn --job_centric
```

```
python3 train.py  --cuda 0 --log_file_desc matnet_10_6_6 --algorithm matnet
```

To test all of the trained models, genetic algorithm and dispatching rules:
```
python3 test.py --multi_test --test_GA --test_dispatch
```

To test bechmark dataset:
```
python3 test_benchmark.py --multi_test --test_GA --test_dispatch
```

To give a comprehensive view of the test results in terms of small and large instances:
```
python3 figure.py
```