# Commenting DLSSVM demo code

Do not use Windows, it's not an OS for CS students.

### Get dataset for demo
```
bash get_dataset.sh
```

### Important
The original script flash the tracking box too fast, you may want to add `pause(T)` into the loop.

### Official Readme.txt
```
(1) DLSSVM tracker without scale estimation.
(2) Scale-DLSSVM handles scale change.
(3) for DLSSVM and Scale-DLSSVM, only tracker.m and makeConfig.m (only config.padding) are different. other files are completely same.

If you have any problem, please contact shaojiejiang@126.com and jf_ning@sina.com.
```
