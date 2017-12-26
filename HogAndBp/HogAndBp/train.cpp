#include <iostream>
#include <fstream>
#include <opencv.hpp>
#include <highgui.h>
#include <cxcore.h>
#include <cstring>
#include <ml.h>
#include <vector>

enum  status{
	TRAIN,
	TEST
}sta;

using namespace std;
using namespace cv;

#define  F_NUM    1764     //7*7*9*4
#define  m_NUM    560
#define  CLASSNUM 4

//----------------------------全局变量定义---------------------------------
	vector<float> descriptors;               //HOG特征存放向量 
	float    data[m_NUM][F_NUM];             //样本特征存放数组
	float    f[1][F_NUM];
	float    dataCls[m_NUM][CLASSNUM];       //样本所属类别
	int      mClass ;                        //训练样本所属类别
	int      dNum;                           //训练样本个数
/*-------------------------------------------------------------------------*/

//-----------------------------函数定义------------------------------------
/**************************************************
*名称：init()
*参数：void
*返回值：void
*作用：初始化各类参数
****************************************************/
void  init()
{
	memset(data,0,sizeof(data));
	memset(dataCls,0,sizeof(dataCls));
	 mClass = -1;
	   dNum = 0;
}

/**************************************************
*名称：getHOG()
*参数：Mat& img
*返回值：void
*作用：获取图像的HOG特征
****************************************************/
void getHOG(Mat& img)
{
	HOGDescriptor *hog = new HOGDescriptor(             
			Size(64,64),      //win_size  检测窗口大小，这里即完整的图
			Size(16,16),      //block_size  块大小
			Size(8,8),        //blackStride 步进
			Size(8,8),        //cell_size  细胞块大小
			9                   //9个bin
			);
	hog -> compute(           //提取HOG特征向量
		img, 
		descriptors,          //存放特征向量
		Size(64,64),            //滑动步进
		Size(0,0)
		);	
	delete hog;
	hog = NULL;
}

/**************************************************
*名称：packData()
*参数：枚举
*返回值：void
*作用：填充特征向量和特征类别
****************************************************/
void packData(status sta)
{
	int p = 0;
	if (sta == TRAIN)
	{
		for (vector<float>::iterator it = descriptors.begin(); it != descriptors.end(); it++)
		{
			data[dNum][p++] = *it;
		}
		dataCls[dNum++][mClass] = 1.0;
	}
	else if(sta == TEST)
	{
		for (vector<float>::iterator it = descriptors.begin(); it != descriptors.end(); it++)
		{
			f[0][p++] = *it;
		}
	}
	//descriptors.clear();
}

/**************************************************
*名称：packData()
*参数：void
*返回值：void
*作用：填充特征向量和特征类别
****************************************************/
int classifier(Mat& image,CvANN_MLP& bp)
{
    

	getHOG(image);
	packData(sta);

	Mat nearest(1, CLASSNUM, CV_32FC1, Scalar(0));	
	Mat charFeature(1, F_NUM, CV_32FC1,f);

    bp.predict(charFeature, nearest);
    Point maxLoc;
    minMaxLoc(nearest, NULL, NULL, NULL, &maxLoc);
    int result = maxLoc.x;
    return result;
}
int main()
{
	init();
	sta = TRAIN;
	ifstream in("trainpath.txt");
	cout<<"2s后开始训练..."<<endl;
	Sleep(2000);
	system("cls");
	string s,ss;
	while( in >> s){
		if(ss != s.substr(0,44)){
			mClass++;            //类别是0，1，2，3
			cout<<mClass<<endl;
		}
		ss = s.substr(0,44);
		 cout<<s<<endl;
//------------------------读入图像，缩放图像----------------------------
        Mat imge = imread(s),img;  
	
	    if(imge.empty())
	    {
			cout<<"image load error!"<<endl;
			system("pause");
			return 0;
	    }
		resize(imge,img,Size(64,64)); 

//------------------------提取HOG特征，放入特征数组---------------------
		getHOG(img);

		packData(sta);        //填充特征数组和类别数组

	}
	
//------------------------建BP神经网络，开始训练------------------------
	CvANN_MLP bp;

	CvANN_MLP_TrainParams params;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,7000,0.001);  //迭代次数7000,最小误差0.001
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;   //训练方法反向传播
	params.bp_moment_scale=0.1;
	params.bp_dw_scale=0.1;
   

	Mat layerSizes = (Mat_<int>(1,3) << F_NUM,48,4 );  //3层神经网络
	Mat trainDate(m_NUM,F_NUM,CV_32FC1,data);
	Mat trainLable(m_NUM,CLASSNUM,CV_32FC1,dataCls);
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);          //激活函数sigmoid
	system("cls");
	cout<<"训练中...";
	bp.train(trainDate,trainLable, Mat(),Mat(), params);  //开始训练
	
	bp.save("carClassifier.xml");
	system("cls");
	cout << "训练完成！！" <<endl;
	cout <<dNum<<endl;
	
//---------------------------------读入图像，开始测试--------------------------
	system("cls");
	sta = TEST;
	cout<<"开始测试..."<<endl;
	Sleep(2000);
	system("cls");
	Mat imge,img;

	ifstream ins("tpath.txt");

	int cls = -1;
	int num=0,c_num=0;
	while( ins >> s){
		memset(f,0,sizeof(f));
		if(ss != s.substr(0,44)){
			cls++;
			cout<<cls<<endl;
		}
		cout<<s<<endl;
		ss = s.substr(0,44);
		imge = imread(s);
		resize(imge,img,Size(64,64));         //使用线性插值
		num++;
		if (classifier(img,bp) == cls)
		{
			c_num++;
		}
	
	}
	system("cls");
	cout<<"测试完成"<<endl;
	cout<<"***************************************"<<endl;
	cout<<"*样本个数："<<num<<endl;
	cout<<"*正确个数："<<c_num<<endl;
	cout<<"***************************************"<<endl;
		system("pause");



}