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
scripts/run_hotel_0.sh
```
Testing result is
```
The annotation performance: sparsity: 10.6281, precision: 47.8882, recall: 59.9924, f1: 53.2612
```
Extracted ratinales are saved in：
```
./beer_results/aspect0/visual_ann.txt
```
Extracted rationale example：
![image](http://note.youdao.com/s/AjhgD75S)

using the others scprit in the `scripts` can obtain the following results.
![image](http://note.youdao.com/s/IBGcSbBL)



