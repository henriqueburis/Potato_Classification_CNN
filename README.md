# Potato Classification Using CNN_Mixup -------> Em Desenvolvimento


By [Luiz H. Buris](http://),  [Fabiana P. Rocha](http://) [Joel F. H. Quispe](http://)


## Introdução

Agricultura inteligente é o termo usado atualmente para uma metodologia de gestão agrícola, que visa a melhor produção possível, com um percentual mínimo de perdas de colheita e lucro máximo sem a necessidade de expansão do território. Dentro deste aspecto, são integrados sensores ou dispositivos que fornecem informações no campo da agricultura, como irrigação, temperatura, colheita, etc. No caso deste projeto, focamos na parte visual que incorpora dispositivos de vídeo ou fotografias, que ajudam a controlar o trabalho da indústria através de algoritmos de visão computacional, como é o caso da classificação de batatas, como é o caso de sabendo que é um tubérculo muito consumido em todo o mundo até hoje; A variedade de batatas que existem pelo mundo são muitas e o que as diferencia uma e outra é tanto a cor ou a forma em que podem ser encontradas, sendo o campo de estudo ainda maior; para o nosso trabalho limitamo-lo apenas a 4 classes já mencionadas.

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

Amostra  |  SAM Teste
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/example_o.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/example_sam.png) 

## Code organization

- `train.py`: .........

## Train
you can now carry out "run" the python scrypt with the following command:

```sh

python3 train.py --train_dir '/train' --test_dir '/test' --lr=0.0001 --seed=202210023 --decay=1e-4 --batch_size 32 --epoch 200

```

## Resultado
 Média 99.75% on test.

## Confusion Matrix 

Fold1   |  Fold2 | Fold3   |  Fold4 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/_Test-confusion_matrix1.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/_Test-confusion_matrix2.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/_Test-confusion_matrix3.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/_Test-confusion_matrix4.png) 



## Graphic Train, Loss, test classification


Train   |  Loss 
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Figure_train.png) |  ![](https://github.com/henriqueburis/Potato_Classification_CNN/blob/main/fig/Figure_loss.png) 



<p align="center">
<img src="./fig/Figure_test_classification.png" width="500px"></img>
</p>
