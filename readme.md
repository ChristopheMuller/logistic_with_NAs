Results of:
* SAEM.OLD (implementation from [wijiang94](https://github.com/wjiang94/misaem))
* SAEM.0L.0A ([new implementation](https://github.com/ChristopheMuller/misaem), no regularization)
* SAEM.001.0A ([new implementation](https://github.com/ChristopheMuller/misaem), some small regularization)

In term of stability:
* SAEM.OLD fails for small n when correlation is high or d is big
* SAEM.0L.0A fails sometimes but raise the right error (error in fit => retry with regularization)
* SAEM.001L.0A never fails

![](tables_and_figures\figures\MCAR_5d_0corr.png)
![](tables_and_figures\figures\MCAR_5d_095corr.png)
![](tables_and_figures\figures\MCAR_20d_05corr.png)