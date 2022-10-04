# Potato Classification Using CNN_Mixup


By [Luiz H. Buris](http://), [Joel F. H. Quispe](http://), [Miguel Sá](http://)


## Introdução

xxxxxxxxx


![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Capturar.PNG)

## Mixup CNN on training

<p align="center">
<img src="./fig/mixup-interpolation.PNG" width="500px"></img>
</p>

## Code organization

- `train.py`: .........



## Train
you can now carry out "run" the python scrypt with the following command:

```sh
python3 train.py --lr=0.0001 --seed=28092022 --decay=1e-4 --batch_size 20 --epoch 200

```

## Resultado
99.16% on test.

## Confusion Matrix 

<p align="center">
<img src="./fig/confusion_matrix_test.png" width="500px"></img>
</p>

## Graphic Train, Loss, test classification


Train   |  Loss 
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Figure_train.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Figure_loss.png) 



<p align="center">
<img src="./fig/Figure_test_classification.png" width="500px"></img>
</p>
