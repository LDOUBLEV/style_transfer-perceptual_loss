# style_transfer-perceptual_loss
This a funny demo of style transfer of paper ：[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
the detail information about this code is in my CSDN [blog](http://blog.csdn.net/qq_25737169/article/details/79192211)

# usage:
## train:
```
python train.py -is_training=True -vgg_model='your vgg model path' -train_data_path='your train dataset' -style_data_path='your style img path' -style_w=100 
```
forexample:
```
python train.py -is_training=True  -vgg_model='your vgg model path'
```

## test
```
python train -is_training=False -test_data_path='your test img' -new_img_name='transfer.jpg' -transfer_model='your saved model after train'
```

# results

