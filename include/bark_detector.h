/*
* bark_detector.h tinyml_mfcc_queue队列的消费者
* 作者：Kend
* 时间：2025.11.26

* 描述：
    消费者，从tinyml_mfcc_queue队列中取出可能是狗吠的pcm数据，调用该模块的函数，返回狗吠的精确pcm数据，以及狗吠标签
* 算法流程：
    1. PSRAM获取出队列的事件数据(PCM)
    2. 对pcm数据进行预处理，包含：滑窗， 填充， 推理， 转float等。
    3. 运行算法

*/