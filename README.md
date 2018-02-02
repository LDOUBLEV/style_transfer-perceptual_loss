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
python train.py -is_training=True  -vgg_model='vgg16.ckpt' -train_data_path='/train2014' -style_data_path='img/wave.jpg' -style_w=100
```
you can download the vgg16.ckpt model from the [url](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)


note : you can change the degree of style transfer by changing style_w value, 
and you can also modify the code to set the args papameter as default


## test
```
python train -is_training=False -test_data_path='your test img' -new_img_name='transfer.jpg' -transfer_model='your saved model after train'
```
forexample:
```
python train.py -is_training=False  -test_data_path='dog.jpg'  -new_img_name='transfer.jpg' -transfer_model='model_saved/wave.ckpt'
```

# results
![](https://github.com/LDOUBLEV/style_transfer-perceptual_loss/blob/master/dog.jpg)
![](https://github.com/LDOUBLEV/style_transfer-perceptual_loss/blob/master/dog-transfer.png)

![](https://github.com/LDOUBLEV/style_transfer-perceptual_loss/blob/master/scene.jpg)
![](https://github.com/LDOUBLEV/style_transfer-perceptual_loss/blob/master/scene-transfer.png)

## note
if you find it is useful for you, please leave your star, thanks. =_=
