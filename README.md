# Distribution Matching Rationalization
这个库是对Distribution Matching Rationalization(DMR)可解释性框架的TensorFlow版本的实现，[Distribution Matching Rationalization]()

# Getting Started
**clone repo**

```
git clone https://dev.rcrai.com/huangyongfeng/distribution-matching-rationalization.git
cd distribution-matching-rationalization
```
**prepare review dataset**


```
mkdir data
cp -r /data1/share/harden/interpret_BERT/DMR_data ./data
mkdir embeddings
cp -r /data1/share/harden/interpret_BERT/embedding/ ./embeddings
```
(1)Reviews Dataset包括beer review dataset与 hotel review dataset，分别对应`data`文件夹下面的`beer_review`和`hotel_review`文件夹；
- `beer_review`中的`aspect`0 1 2分别对应appearance, aroma, and palate；
- `hotel_review`中的`aspect`0 1 2分别对应location, service, and cleanliness；


(2) `embeddings`中存放的是词向量文件，我们采用的是glove词向量。

**build environment**

```
pip install -r requirements.txt
```

**training**
以beer reviews dataset的aspect=0为例，跑如下脚本：
```
#beer review dataset, aspect=0 appearance
scripts/run_beer_0.sh
```
预期结果为
```
The annotation performance: sparsity: 11.6997, precision: 83.5609, recall: 52.8048, f1: 64.7144
```
对annotation test dataset的rationale预测结果储存于：
```
./beer_results/aspect0/visual_ann.txt
```
预测的rationale示例如下：
![image](https://note.youdao.com/yws/api/personal/file/AFFD88943D144AD7BA54ECBE0BB11E8A?method=download&shareKey=1e6e1b421e3faa4b9c7f874ec963ea2e)

运行`scripts`中的其他脚本可以得到如下的结果
- beer review
![image](https://note.youdao.com/yws/api/personal/file/F8A2A4DB18A3441BAAC8B3E196BF21ED?method=download&shareKey=1637a8956a1a906f66998eae30700443)
- hotel review
![image](https://note.youdao.com/yws/api/personal/file/A470599E7C994AFEBBF8C41B5E455CBF?method=download&shareKey=99f3d61d9ad47f5784785047ce9f0cda)




