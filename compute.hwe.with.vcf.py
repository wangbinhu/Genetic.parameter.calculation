import allel
import pandas as pd
import numpy as np
from scipy.stats import chi2

# 定义输入的VCF文件名和输出的CSV文件名
vcf_filename = 'hn.s120.str.renamed.ID.vcf.gz'
output_csv_filename = 'hwe_results_for_str.csv'

# 读取VCF文件
callset = allel.read_vcf(vcf_filename, fields=['calldata/GT', 'variants/POS', 'variants/CHROM'])

# 初始化结果列表
hwe_results = []

# 定义计算HWE的函数
def calculate_hwe_chi2(gt_data):
    # 计算每个等位基因的总计数
    n_alleles = np.max(gt_data) + 1
    allele_counts = np.zeros(n_alleles, dtype=np.int64)
    for sample in gt_data:
        for allele in np.ravel(sample):  # 将每个样本的基因型展平
            if allele >= 0:  # 忽略缺失数据
                allele_counts[allele] += 1

    # 总的等位基因数
    total_alleles = np.sum(allele_counts)

    # 计算等位基因频率
    allele_freqs = allele_counts / total_alleles

    # 初始化预期的基因型计数
    expected_counts = np.zeros((n_alleles, n_alleles), dtype=np.float64)
    for i in range(n_alleles):
        for j in range(i, n_alleles):
            exp_freq = allele_freqs[i] * allele_freqs[j] * total_alleles
            if i == j:
                expected_counts[i, j] = exp_freq
            else:
                expected_counts[i, j] = expected_counts[j, i] = exp_freq * 2

    # 初始化观察到的基因型计数
    observed_counts = np.zeros_like(expected_counts, dtype=np.int64)
    for sample in gt_data:
        a1, a2 = sample
        if a1 >= 0 and a2 >= 0:
            observed_counts[a1, a2] += 1
            if a1 != a2:
                observed_counts[a2, a1] += 1

    # 计算卡方统计量
    with np.errstate(divide='ignore', invalid='ignore'):
        nonzero_expected = expected_counts > 0
        chi2_stat = np.nansum((observed_counts[nonzero_expected] - expected_counts[nonzero_expected]) ** 2 / (expected_counts[nonzero_expected] + 1e-8))

    # 自由度
    df = (n_alleles * (n_alleles - 1) / 2) - 1

    # 计算P值
    p_value = chi2.sf(chi2_stat, df)

    return round(p_value, 2)  # 四舍五入到小数点后两位

# 遍历位点
for i in range(len(callset['variants/POS'])):
    chrom = callset['variants/CHROM'][i]
    pos = callset['variants/POS'][i]
    chrom_pos = f'{chrom}_{pos}'

    gt_data = callset['calldata/GT'][i]

    # 计算HWE p值
    hwe_p_value = calculate_hwe_chi2(gt_data)

    # 添加结果到列表
    hwe_results.append([chrom_pos, hwe_p_value])

# 保存结果到CSV
df_hwe = pd.DataFrame(hwe_results, columns=['STR', 'HWE_p_value'])
df_hwe.to_csv(output_csv_filename, index=False)

# 打印完成信息
print(f'Results saved to {output_csv_filename}')
