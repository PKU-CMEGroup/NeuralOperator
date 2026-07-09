# 操作指南
首先因为是从别的文件夹中迁移过来，可能难免有文件夹错误等问题，或许需要额外的修补。

### Step 1

下载好1~500(实际共484个.vtp数据)，放在`data/DrivAerML/data`下。
data/hifi3d_processed/test
### Step 2

在`scripts/drivaerml`文件夹下运行(或者提交sbatch)
```sh
bash decimate.sh
```
于是应当产生了`data/HiFi3D/DrivAerML_20000`文件夹

### Step 3

在`scripts/drivaerml`文件夹下运行(或者提交sbatch)
```sh
bash preprocess.sh
```

于是应当产生了`data/hifi3d_processed/test`文件夹其中有一个.npz和一个.npy

### Step 4

运行`scripts/drivaerml`文件夹里面的训练脚本，可以在.sh脚本中修改参数。