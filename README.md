# HRankFL

## 开始
+ localSet目录存放内容
+ results/vgg目录存放内容

## 架构

+ model：网络模型实例——通道调整创建、原型模型实例类变量
+ pruningInfo：剪枝控制信息实例——提供控制信息计算更新、模型密度获取
+ vdevice：设备控制实例（封装model）——模型和张量的加载、模型序列化与反序列化
+ vwrapper：运行时实例（封装vdevice）——训练时损失函数和优化函数、训练和测试、运行时配置保存、模型训练开销测算
+ vhrank：剪枝算法实例（封装vwrapper）——算法运行、模型参数更新、通信参数规范
+ node：运行结点实例（封装vhrank）——装配loader和算法实例、通信交互

```mermaid
flowchart TB
	c1-->|不同子图元素<br/>之间的连接| a2
    subgraph one
    a1-->|子图内部元素<br/>之间的连接| a2
    end
    
    subgraph two
    b1-->b2
    end
    
    subgraph three
    c1-->c2
    end
    
    one -->|子图之间的连接| two
    three --> two
    two --> c2
```



## 开发日志

### 待实现

+ argParser & runtimeEnv：参数解析器与执行环境的绑定

+ pathHandler：路径绑定与加载器

+ vdevice：模型的序列化与反序列化

+ vwrapper：模型训练时配置加载与保存，模型训练时开销测算

+ dynamicChange：网络模型结构固化或是动态变化

+ hyperProvider：剪枝参数自适应给出

+ model：原型模型参数类变量创建

+ autoParser：Json配置与解析

+ 注释补充与函数接口完善

+ args&runtimeEnv：command和json配置协调

  

+ fltrain：实现联邦学习过程

+ netComm：实现真实网络交互通信

+ ext：实现ResNet、MobileNet在ImageNet上的效果



## 待优化

1. runtimeEnv和preEnv需要进一步整合区分
2. 修改枚举类型的传入和合法判断
3. 添加注释，修改命名合乎规范，内部类前缀_
4. 封装test_unit.py接口
5. 重新梳理各个模块的耦合，封装易用的抽象类



## 难点

+ 剪枝参数的确定，是一个超参数
+ 为什么要判断learning rate一样
+ 怎样保证master节点和worker节点上的模型都收敛



## 原则

1. test_unit.py可提供单元测试或是对外接口

2. 每个包对外提供的接口只保留在test_unit.py中，所有外部包只调用该py包的接口获得服务

   

## 需求

+ 给定一个损失精度-5%，然后求得最小网络子结构，获得网络中各卷积层的通道数



## 不合理

+ datasettype datatype
+ data_per_client = local_update_epoch * batch_size



## 依赖库

+ torch, torchaudio, torchvision
+ DecDeprecated, fedlab





## 本机调试支持

+ preEnv 10：

gpu=0



+ vwrapper 157：

checkpoint = torch.load(path, map_location='cuda:0')



+ message 59-60：

\# kwargs['alg'].get_rank(kwargs['loader'])

kwargs['alg'].deserialize_rank()



+ vdevice 161：

self.model.load_state_dict(adapt_dict, False)





## 优秀的开发

+ 提供健壮的接口
+ 提供合适的扩展编写方式
