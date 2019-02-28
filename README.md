# shiFu
## 代码架构

```shell
shiFu/ # 我们的github1项目    
    my_project/ # 存放我们的项目代码
    	...
    	model/ # 存放模型.py
    		...
    baseline/ # 已经公开的源代码
        FM/
        xDeepFM/
    export/ # 数据探索&可视化(存放jupyter或者html数据探索文件)
data/
	track1/ # 第一个赛道的数据，大约40G
		... # 各种csv(原始数据)
	track2/ # 第二个赛道的数据，大概5G
		...
cache/
	track1/
        checkpoints/ # 用来存放模型文件
        runs/ # 用来存在运行的记录（比如说tensorboard可视化文件）
        result/ # 存放提交的csvfile
        tmp/ # 存放data_process.py分离的train, val数据
        	...
        	hd5/ # 存放hd5大文件
    track2/
    	checkpoints/ # 用来存放模型文件
        runs/ # 用来存在运行的记录（比如说tensorboard可视化文件）
        result/ # 存放提交的csvfile
        tmp/ # 存放data_process.py分离的train，val数据
        	...
        	hd5/ # 存放hd5大文件
```

## 如何运行

**step one**

查看`config.py`文件，选择运行`track1`还是`track2`的数据。

**step two**

```python
python data_process.py # 切分成train-val（8:2），生成hd5文件
```

**step three**

```python
python train.py
```

