## 本程序基本ubuntu18.04依赖tensorrt库和opencv库，请自行安装，安装后将对应的头文件和库文件放入include和lib文件夹下

## mytrt目录实现将原本的tensorrt封装成一个动态库，封装了将onnx转换成tensorrt所需要的engine，forward，数据在设备端拷贝功能。
## openpose 实现了openpose的tensorrt 包括前处理和后处理
## pose实现了mmpose的tensorrt前向传播并包含图片的前后处理过程
## yolov5实现了yolov5的tensorrt前向传播包含前后处理过程
### 注意修改每个目录下test目录下图片和模型的路径
### [相关onnx模型下载](https://download.csdn.net/download/qq_34929889/85407018?spm=1001.2014.3001.5503)
