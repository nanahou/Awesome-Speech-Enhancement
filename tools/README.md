# Start to use the tool

## For PESQ Measurement

Please run the script below to download the widely used toolkit.

```bash
sh download_pesq_tool.sh
```

Scripts download 2 PESQ packages, the calculation results of two packages are different due different versions. Please decide which one is prefered to you.

To evaluate the difference between calculations between two packages, we utilize the public testset of [dataset by University of Edinburgh](https://datashare.is.ed.ac.uk/handle/10283/1942). The following results are calculated between "noisy_testwav" and "clean_test_wav".

|Package|PESQ|CSIG|CBAK|COVL|SegSNR|
|----|----|----|----|----|----|
|COMPOSITE|3.02|3.99|2.95|3.48|1.74|
|K14513_CD_Files|1.97|3.35|2.44|2.63|1.68|

## More Tools is Coming Soon.

^_^