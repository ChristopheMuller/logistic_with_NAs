Results of:
* SAEM.R.OLD (implementation from [wijiang94](https://github.com/wjiang94/misaem))
* SAEM.R.NoReg ([new implementation](https://github.com/ChristopheMuller/misaem), no regularization: lambda = 0, alpha = 0)
* SAEM.R.Reg ([new implementation](https://github.com/ChristopheMuller/misaem), some small regularization: lambda = 0.01, alpha = =1=Ridge)
* SAEM.PY.NoReg ([new implementation](https://github.com/ChristopheMuller/misaem_python)), no regularization
* SAEM.PY.Reg ([new implementation](https://github.com/ChristopheMuller/misaem_python)), some regularization (l2, C=1)

In term of stability:
* SAEM.R.OLD fails for small n when correlation is high or number of covariates is big
* SAEM.R.NoReg fails sometimes but raise the right error (suggest to add regularisation)
* SAEM.R.Reg never fails
* SAEM.PY.Reg and SAEM.PY.NoReg never fails

![mcar5d0c](tables_and_figures/figures/MCAR_5d_0corr.png)
![mcar5d095c](tables_and_figures/figures/MCAR_5d_095corr.png)
![mcar20d05c](tables_and_figures/figures/MCAR_20d_05corr.png)
