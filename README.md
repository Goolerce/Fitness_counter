# Fitness_counter
基于BlazePose+KNN实现人体姿态健身计数
项目描述：
实现基于mediapipe的人体姿态识别的AI健身自动计数功能
支持健身动作：1、俯卧撑	2、深蹲	      3、引体向上  	 4、仰卧起坐
创建时间：2022.11.28
完成时间：2022.11.28
联系邮箱：YanglongLiu@163.com

如何训练新的健身动作模型？
1、修改mian函数

2、首先在fitness_pose_images_in的文件夹下存储对应健身的初态动作与末态动作图像

3、修改videoprocess.py文件中的代码，flag模式选择部分，注意class_name必须与fitness_pose_images_in文件夹下的文件名字保持一致

4、修改videoprocess.py文件中的代码，flag模式选择部分，注意class_name必须与fitness_pose_images_in文件夹下的文件名字保持一致

5、修改trainingsetprocess.py文件中的代码，flag模式选择部分，注意 文件名 必须与fitness_pose_images_in文件夹下的文件名字保持一致
