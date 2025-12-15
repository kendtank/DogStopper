// 模版管理 + Flash 持久化


/**
 * 
 * 它解决什么问题？

模版的持久化

计数（学了多少次）

上电恢复

它不关心

embedding 从哪来

聚类算法是什么

相似度怎么算
 */

 // template_storage.h
bool template_storage_load(float* out_embed, int* out_count);
bool template_storage_save(const float* embed, int count);
