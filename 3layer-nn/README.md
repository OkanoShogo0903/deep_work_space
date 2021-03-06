# 3層ニューラルネットワーク
![plot_box_eta03](https://user-images.githubusercontent.com/25472671/50591783-8fcdde00-0ed4-11e9-971b-90219cb22e1d.png)

- 導入,手法  
**3層ニューラルネットワークによってニューヨークにおける地価予測**を回帰問題として扱った.  
中間層の活性化関数にはシグモイド関数を用いて、出力層は回帰問題であるために活性化関数を用いなかった.  
- 結果  
結果として、**分散5.0ドル、標準偏差7.1ドル**の精度で地価予測を行うことができた.  
また、500個のデータセットに対してEpoch数を増やしても精度は向上しなかったが、学習率を増やすことによって精度が向上したことを確認した.  
- 考察  
これは小さいデータセットに対しては**学習率**を`0.1 ~ 0.5`のように大きな値を設定することにより、良い結果を得ることができるのではないかと考える.

## TODO
- etaを0.5にすると、nanがでてきて学習がうまくいかない

## Install
```
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data

```

## Using
```
$ python nn.py
```

## HotReference
```
    ''' return 0~1
    >>> sigmoid( np.array([1, 3]), np.array([2, 1]))
    '''
```
```
numpy.seterr(all=None, divide=None, over=None, under=None, invalid=None)
```
## Epoch数についての考察
### Epoch 1
```
mean : -5.204653014437471
var  : 18.134327272727273
std  : 4.2584418832158875
```

### Epoch 3
```
mean : -5.154306823483189
var  : 18.134327272727273
std  : 4.2584418832158875
```

## 学習率(Eta)についての考察
### Eta 0.05
```
mean : -5.187206071452584
var  : 18.134327272727273
std  : 4.2584418832158875
```

### Eta 0.1
```
mean : -5.520000047780304
var  : 18.134327538677802
std  : 4.258441914442159
```

### Eta 0.5
```
mean : 1.660704401679474e+147
var  : 5.038209258419659e+263
std  : 7.098034416949286e+131
```
```
mean : -5.308268300899038e+80
var  : 4.436271510593304e+130
std  : 2.1062458333711437e+65
```
