#include <iostream>
#include <fstream>
#include <opencv.hpp>
#include <highgui.h>
#include <cxcore.h>
#include <cstring>
#include <ml.h>
#include <vector>
#include <iomanip>
//#define TRAIN 
#define TEST
using namespace std;
using namespace cv;


//------------------------ 全局变量-----------------------------------------
#define  F_NUM     1764     //7*7*9*4  车标特征维数
#define  N_SAMPLE  1000     //训练样本个数
#define  CLASSNUM  5        //车标种类 
float Data[N_SAMPLE][F_NUM];
float Label[N_SAMPLE][CLASSNUM];
vector<float> descriptor;   // HOG特征存放容器

//-----------------------------全局函数------------------------------------------
// 获取图像hog特征
void getHOG(Mat& img) {
	HOGDescriptor *hog = new HOGDescriptor(
		Size(64, 64),      //win_size  检测窗口大小，这里即完整的图
		Size(16, 16),      //block_size  块大小
		Size(8, 8),        //blackStride 步进
		Size(8, 8),        //cell_size  细胞块大小
		9                   //9个bin
	);
	hog->compute(           //提取HOG特征向量
		img,
		descriptor,          //存放特征向量
		Size(64, 64),            //滑动步进
		Size(0, 0)
	);
	delete hog;
	hog = NULL;
}

// 切分字符串
vector<string> split(string s, char token) {
	istringstream iss(s);
	string word;
	vector<string> vs;
	while (getline(iss, word, token)) {
		vs.push_back(word);
	}
	return vs;
}

// 装填数据
void packTrainData(Mat &img, int label, int counter) {
	getHOG(img);// 获取图片HOG特征
	int cur = 0;
	for (auto d : descriptor)
		Data[counter][cur++] = d;
	Label[counter][label] = 1.0;
}

//------------------------------网络类--------------------------------------------
class myNeuralNetwork {
public:
	myNeuralNetwork() {};
	myNeuralNetwork(char *str) { this->bp.load(str); }
	void initialNN();     // 初始化网络参数
	void train(float(&data)[N_SAMPLE][F_NUM], float(&label)[N_SAMPLE][CLASSNUM]);  //训练
	int  predict(Mat &img);
private:
	CvANN_MLP_TrainParams params;  // 网络参数
	CvANN_MLP bp;		// bp网络
};

void myNeuralNetwork :: initialNN() {
	//term_crit终止条件 ，它包括两项，迭代次数和误差最小值(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS),一旦有一个达到条件就终止
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 0.001);  //迭代次数7000,最小误差0.001

	//train_method训练方法,opencv里面提供了两个方法一个是经典的反向传播算法BP,一个是弹性反馈算法RPROP
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;   //训练方法反向传播

	//bp_moment_scale权值更新冲量
	params.bp_moment_scale = 0.1;
	//bp_dw_scale权值更新率
	params.bp_dw_scale = 0.1;

	Mat layerSizes = (Mat_<int>(1, 3) << F_NUM, 48, CLASSNUM);  //3层神经网络
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);          //激活函数sigmoid
}

void myNeuralNetwork::train(float(&data)[N_SAMPLE][F_NUM], float(&label)[N_SAMPLE][CLASSNUM]) {
	Mat trainData(N_SAMPLE, F_NUM, CV_32FC1, data);
	Mat trainLabel(N_SAMPLE, CLASSNUM, CV_32FC1, label);
	bp.train(trainData, trainLabel, Mat(), Mat(), params);  //开始训练
	bp.save("carClassifieryu.xml");
}

int myNeuralNetwork::predict(Mat &img) {
	::getHOG(img);
	float testData[1][F_NUM];
	int cur = 0;
	for (auto d : descriptor) 
		testData[0][cur++] = d;

	Mat nearest(1, CLASSNUM, CV_32FC1, Scalar(0));
	Mat charFeature(1, F_NUM, CV_32FC1, testData);
	bp.predict(charFeature, nearest);
	Point maxLoc; minMaxLoc(nearest, NULL, NULL, NULL, &maxLoc);
	int result = maxLoc.x;
	return result;
}

int main() {
#ifdef TRAIN
	ifstream in("trainpath.txt");
	string s;

	int label;
	int counter = 0;
	while (in >> s) {
		// 获取label
		vector<string> fp = split(s, '/');
		label = fp[3].c_str()[0] - '0';
		// 获取图片信息
		Mat imge = imread(s), img;
		if (imge.empty())
		{
			cout << "image load error!" << endl;
			system("pause");
			return 0;
		}
		resize(imge, img, Size(64, 64));
		packTrainData(img, label,counter++);
	}
	cout << "加载训练集完毕!" << endl;
	cout << "开始训练........." <<endl;
	myNeuralNetwork ann;
	ann.initialNN();
	ann.train(Data, Label);
	cout << "训练完成！" << endl;
#endif
#ifdef TEST
	ifstream test_in("testpath.txt");
	string ss;
	int true_label;
	int counter = 0;
	int wrong = 0;
	string carlog[5] = { "雪铁龙","大众","一汽","福田","本田" };
	cout << "开始测试..." << endl;
	cout << "进度: 0.00%";
	while (test_in >> ss) {
		// 获取label
		vector<string> fp = split(ss, '/');
		true_label = fp[3].c_str()[0] - '0';
		// 获取图片信息
		Mat imge = imread(ss), img;
		if (imge.empty())
		{
			cout << "image load error!" << endl;
			system("pause");
			return 0;
		}
		resize(imge, img, Size(64, 64));
		myNeuralNetwork predictor("carClassifieryu.xml");
		int predict_label = predictor.predict(img);
		if (predict_label != true_label)
			wrong++;
		counter++;
		cout<<"\r进度: "<< setprecision(2) << fixed << counter*1.0 / 5 << "%" ;

		
	}
	cout << endl;
	cout << "----------------------" << endl;
	cout << "测试结果" << endl;
	cout << "----------------------" << endl;
	cout << "目标数目：" << counter << endl;
	cout << "正确个数：" << counter-wrong << endl;
	cout << "正确率：" << setprecision(2)<<fixed<<(counter-wrong)*1.0/counter*100 <<"%"<< endl;
	cout << "----------------------" << endl;
#endif
	system("pause");
	return 0;
}