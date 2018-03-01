# carLogoRec
### opencv自带BP网络+Hog特征识别车标
opencv2.4.11：使用HOG特征进行车标分类识别

>工程基于本人早期课程设计，记录在博客园博客中---[使用OpenCV实现车标识别](http://logwhen.cn/2018/01/16/%E4%BD%BF%E7%94%A8OpenCV%E5%AE%9E%E7%8E%B0%E8%BD%A6%E6%A0%87%E8%AF%86%E5%88%AB.html)。 Github上的版本是重构后的代码。

### 环境要求
测试环境：win10+VS2017+OpenCV2.4.11  
测试结果：100%识别  
opencv的配置这里就不再赘述了。简单讲下代码的情况。  
### 代码概览
代码很简单，就一个`main.cpp`文件。大致分为以下3块
- **全局变量**：比如整理好的数据集，标签集，HOG特征向量 
- **全局函数**：模块划分，使得`main`函数不显得臃肿。
- **自定义网络类**：`myNeraulNetwork`用于搭建简单BP网络和训练预测

### 训练集&测试集准备
1. 先将数据集手动划分成训练集和测试集，并分好类，比如第一类就放在文件夹名为`0`的文件夹下，第二类就是`1`，如此类推。
>当前程序只能处理10类以下车标，因为当前程序逻辑不支持10以上的数字识别（具体可以仔细看下代码）

 所有训练集的图片放在`train`文件夹中，测试集放在`test`文件夹下。最终的文件树如下：
![捕获.PNG](https://i.loli.net/2018/03/02/5a982f6d2826a.png)
> 1. `reCarlog`是工程名，即`Cardata`和`main.cpp`同目录。
> 2. 测试集的类别数字和训练集的要一一对应。因为程序将要用它们作为分类依据。

2. 在 `main.cpp`目录下准备两个文件，`trainpath.txt`和`testpath.txt`，用以保存所有训练集和测试集图片的路径。举例如下(trainpath.txt)
> ./Cardata/train/0/train_citroen1.jpg
> ./Cardata/train/0/train_citroen10.jpg
./Cardata/train/0/train_citroen100.jpg
./Cardata/train/0/train_citroen101.jpg

	建议使用相对路径。

