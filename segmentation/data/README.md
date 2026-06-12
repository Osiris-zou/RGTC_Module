# ADE20K preparation

Download ADE20K from its official provider. Expected layout:

```
ADEChallengeData2016/
  images/validation/*.jpg
  annotations/validation/*.png
```

The paper evaluates all 2,000 validation images with 150 classes, short side 512, maximum long side 2048, sliding window 512, stride 480, and window batch size 4.
