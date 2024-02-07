利用python计算多等位基因STR.vcf.gz 哈代-温伯格平衡（HWE）的P值

此脚本首先读取VCF文件，遍历每个位点的基因型数据，计算每个位点的哈代-温伯格平衡检验的P值，并将结果保存到一个CSV文件中。
通过在计算卡方统计量时添加一个非常小的数值（1e-8）到预期计数的分母，避免了除以零的问题。此外，P值结果被四舍五入到小数点后两位，以便于阅读和解释。
