# shiFu
## 代码架构

```shell
shiFu/ # 我们的github1项目    
    my_project/ # 存放我们的项目代码
    baseline/ # 已经公开的源代码
        FM/
        xDeepFM/
    export/ # 数据探索&可视化(存放jupyter或者html数据探索文件)
data/
	track1/ # 第一个赛道的数据，大约40G
		...
	track2/ # 第二个赛道的数据，大概5G
		...
cache/
	checkpoints/ # 用来存放模型文件
	runs/ # 用来存在运行的记录（比如说tensorboard可视化文件）
```

​	