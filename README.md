#BIT图像视频处理大作业


----

## 环境

1. **python:** python 3.6.9
2. **python-opencv:** 3.4.2
3. **system:** windows10

**说明:** 以下代码皆为自己编写完成，opencv 仅用于图像的读取和保存, 该代码已放在github上， 地址为:
, 用户在使用时可以使用自己的图像，图像最好放在```img```目录下， 执行命令后对应的结果文件存入```result```对应的目录中。

----

## 第一题:直方图均衡化

### 运行命令

```python histogramEqualization.py --img img\car.jpg --Level 256```

**参数:**

**img:** 该参数用于指定需要处理的图片，默认的示例相对路径为```img\car.jpg```

**Level:** Level 用于指定 L 灰度级

### 运行结果

除了使用```matplotlib```包中```plt.imshow()```进行数据展示外, 处理后的图像保存在相对路径```result\His\```目录下。

----

## 第二题： 小区域面积

### 运行命令

```python main.py```

**参数:**

这里因为是实现一个区域的算法，为了简化并没有使用外部的图片，采用的是自己定义的一个矩阵，只需要运行代码即可。

----

## 第三题:链码

### 运行命令

```python chainCode.py --img img\three.jpg --threshold 70```

**参数:**

**img:** 该参数用于指定需要处理的图片，默认的示例相对路径为```img\three.jpg```

**threshold:** threshold 用于指定 opencv 二值化的阈值, 像素点大于该阈值会被置为255, 小于该阈值会被置为0

### 运行结果

该代码主要使用```plt.imshow()```和```plt.plot()```展示链码划分的边缘，没有保存最后的结果图片，仅显示。

### 注意事项

本次图片使用了较为简单的手写数字识别图片，在```img\train.csv```中每行对应一张图片的784维像素值和标签，但是将这些像素点使用opencv保存图片再读取后会出现一些像素点的值变化，因此会导致结果不是很好，如果想展示较好结果，可以修改代码直接从```img\train.csv```中读取一行进行展示。

----


## 第四题: 频域滤波

### 运行命令

频域滤波实现了**频域锐化，高斯高通滤波器**和**频域平滑，高斯低通滤波器**两种:

**高斯高通:**

```python frequencyDomainFiltering.py --img img\car.jpg --d 10 --mode high```

**高斯低通:**


```python frequencyDomainFiltering.py --img img\car.jpg --d 10 --mode low```

**参数:**

**img:** 该参数用于指定需要处理的图片，默认的示例相对路径为```img\car.jpg```

**d:** 高斯滤波器传输函数中的D0阈值

**mode:** 高通或低通模式

### 运行结果

除了使用```matplotlib```包中```plt.imshow()```进行数据展示外, 高通处理后的图像保存在相对路径```result\Frequency\high\```目录下, 低通处理后的图像保存在相对路径```result\Frequency\low\```目录下。

----

## 第五题: 拉普拉斯图像增强

### 运行命令

拉普拉斯图像增强采用了两种不同的拉普拉斯滤波器, 用户可以通过```mode```参数选择:

	sharp: [[0,1,0],[1,-4,1],[0,1,0]]滤波器
	 edge: [[1,1,1],[1,-8,1],[1,1,1]]滤波器

**sharp模式:**

	python laplacian.py --img img\car.jpg --mode sharp

**edge模式:**

	python laplacian.py --img img\car.jpg --mode edge

**参数:**


**img:** 该参数用于指定需要处理的图片，默认的示例相对路径为```img\car.jpg```


**mode:** 选择不同的滤波器矩阵

### 运行结果

除了使用```matplotlib```包中```plt.imshow()```进行数据展示外, 拉普拉斯图像增强 sharp模式处理后的图像保存在相对路径```result\Lap\sharp\```目录下, 拉普拉斯图像增强 edge模式处理后的图像保存在相对路径```result\Lap\edge\```目录下。

----










