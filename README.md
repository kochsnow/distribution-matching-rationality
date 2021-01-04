# Distribution Matching Rationalization
This repo contains Tensorflow implementation of Distribution Matching Rationalization(DMR)，[Distribution Matching Rationalization]()

# Getting Started
**prepare review data**
For the original [beer review dataset](https://snap.stanford.edu/data/web-BeerAdvocate.html) has been removed by the dataset’s original author, at the request of the data owner, BeerAdvocate. We use [hotel review dataset](https://people.csail.mit.edu/yujia/files/r2a/data.zip) in the demo.
To prepare data, runing the scripts as the following
```
sh download_data.sh
```


**build environment**
```
pip install -r requirements.txt
```

**training**
Taking appearance aspect of beer review dataset as example
```
#beer review dataset, aspect=0 appearance
scripts/run_beer_0.sh
```
Testing result is
```
The annotation performance: sparsity: 11.6997, precision: 83.5609, recall: 52.8048, f1: 64.7144
```
Extracted ratinales are saved in：
```
./beer_results/aspect0/visual_ann.txt
```
Extracted rationale example：
![image](https://note.youdao.com/yws/api/personal/file/AFFD88943D144AD7BA54ECBE0BB11E8A?method=download&shareKey=1e6e1b421e3faa4b9c7f874ec963ea2e)

using the others scprit in the `scripts` can obtain the following results.
- beer review
![image](https://note.youdao.com/yws/api/personal/file/F8A2A4DB18A3441BAAC8B3E196BF21ED?method=download&shareKey=1637a8956a1a906f66998eae30700443)
- hotel review
![image](https://note.youdao.com/yws/api/personal/file/A470599E7C994AFEBBF8C41B5E455CBF?method=download&shareKey=99f3d61d9ad47f5784785047ce9f0cda)




