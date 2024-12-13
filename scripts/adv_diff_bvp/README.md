# Data Information

Download advection diffusion boundary value problem  data from 

PKU drive
https://disk.pku.edu.cn/link/AR082433050E2C4D4E9B31E54728B02B4A
Name: adv_diff_bvp.zip

File Location: AnyShare://Neural-Operator-Data/adv_diff_bvp.zip

# tips for training L

Select 'train_sp_L' from  False , 'together' and 'independently':
## False
means that L will not be trained;  
## 'together' 
means that L will be trained with other params in the same optimizer;  
## 'independently' 
means that L will be trained independently in another optimizer with a new learning rate, which is equal to  base_lr * lr_ratio;     


usually, training process will be more stable with a fixed L , but it will have more chance to generalize with L trained.

I recommend setting train_sp_L to 'independently' with an appropriate lr_ratio (a ratio of 10 might be suitable). 
When significant fluctuations occur and the model performance consistently worsens, you'd better decrease the lr_ratio or reconsider the type of train_sp_L.