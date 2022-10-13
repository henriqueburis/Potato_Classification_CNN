# Potato Classification Using CNN_Mixup -------> Em Desenvolvimento


By [Luiz H. Buris](http://), [Joel F. H. Quispe](http://), [Miguel Sá](http://)


## Introdução

xxxxxxxxx


![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Capturar.PNG)


## Citation

If you use this method or this code in your paper, then please cite it:

```
@article{XXXXXXXXXX,
  title={XXX},
  author={Luiz H Buris, Joel F. H. Quisp, Miguel Sá},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={20XX},
  url={https:https://arxiv.org/pdf/XXXX.XXXXX.pdf},
}
```

## Mixup CNN on training

<p align="center">
<img src="./fig/mixup-interpolation.PNG" width="500px"></img>
</p>

## Image segment classification on train model.


hsv,frame


Frame Teste  |  hsv Teste
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/frameClone.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/blur.png) 




## Code organization

- `train.py`: .........



## Train
you can now carry out "run" the python scrypt with the following command:

```sh

python3 train.py --train_dir '/train' --test_dir '/test' --lr=0.0001 --seed=202210023 --decay=1e-4 --batch_size 32 --epoch 200

```

## Resultado
 99.67% on test.

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
