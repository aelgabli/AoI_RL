#Installation
To install the dependencies, run
```
python setup.py
```


#Training
To train a model, go inside train directory and run 
```
python Final_code.py
```

The training process can be monitored in `sim/results/log_test` (validation) and `sim/results/log_central` (training). 

Trained model will be saved in `sim/results/`. 

#Testing
Trained RL model needs to be copied to `test/models/`. 

To test a trained model for the proposed solution, go inside test directory and run 
```
python proposed.py
```
To test baseline 1, run
```
python base1.py
```
To test baseline 2, run
```
python base2.py
```
#Plotting Results
Results will be saved in `test/results/`. 

To view the results, run 
```
python plot_results.py
```
#Citation

@article{elgabli2018reinforcement,
  title={Reinforcement learning based scheduling algorithm for optimizing age of information in ultra reliable low latency networks},
  author={Elgabli, Anis and Khan, Hamza and Krouka, Mounssif and Bennis, Mehdi},
  journal={arXiv preprint arXiv:1811.06776},
  year={2018}
}

# AoI_RL
