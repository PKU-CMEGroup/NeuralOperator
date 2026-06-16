# Code Structure

## 1) Generate dataset
```bash
python generate_burgers1d_data.py
```
The script supports periodic data with uniform grid initialized with Gaussian random field

## 2) Train a 1D neural operator, 
Train a Fourier Neural Operator (FNO) to learn the map $(\u(0) x) \rightarrow \u(t)$ 
```bash
python fno_train.py
```
Train a PCNO to learn the same mapping. The length scale for Fourier feature is $2.1L$, this setting does not leverage periodicity  
Firstly, use the preprocess_data script to preprocess the PCNO data. In the `preprocess_data` function, turn on `preprocess_data` for preprocessing PCNO data. 
```bash
python pcno_train.py
```
Train a PCNO to learn the same mapping, the setting leverage periodicity, and the PCNO data is generated on the fly 
```bash
python pcno_periodic_train.py
```

## 3) Test and roll out (autoregressive)
Evaluate the trained model on $\u(0) \rightarrow \u(t)$ then roll the prediction forward for 100 steps:
```bash
python fno_plot_results.py
python pcno_plot_results.py
python pcno_periodic_plot_results.py
```


# Observations
50 iterations
## 1) FNO: Time:  7.884  Rel. Train L2 Loss :  0.0071429064565338195  Rel. Test L2 Loss :  0.009618622669018804  100 iteration 0.49350023

## 2) PCNO periodic (no smooth): Time:  29.122  Rel. Train L2 Loss :  0.004451535734860227  Rel. Test L2 Loss :  0.006468603378161788,   100 iteration 0.38425426

## 3) PCNO: Time (no smooth):  29.957  Rel. Train L2 Loss :  0.005763772659469396  Rel. Test L2 Loss :  0.01050229353737086,   100 iteration 1.4657369562 