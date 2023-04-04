# DoS/DDoS Anomaly Detection
The three goals of a secure network are confidentiality, integrity, and availability. DoS (Denial of Service)/DDoS (Distributed Denial of Service) attacks are attacks that aim to make data, hosts, networks, or other online resources unavailable to legitimate users through the manipulation of multiple sources. These attacks can be catastrophic to enterprise networks if these networks do not have cost-effective and timely detection solutions to detect and mitigate these attacks. In previous works, various machine learning methodologies have been proven instrumental in attack detection. In this paper, we compared and implemented machine learning algorithms, including multiple linear regression, decision tree, and support vector machine, to realize a DoS/DDoS hybrid intrusion detection system based on two well-known datasets: KDD-CUP 1999 and CICIDS-2017. The performance of each algorithm is compared and analyzed. We propose a DoS/DDoS hybrid intrusion detection system which integrates various machine learning algorithms for best efficiency. In specifics, the system is composed of three key parts: feature reduction, network anomaly detection, and signature-based classification. As a result, a classification accuracy score of 99.98% is reached for both datasets.

# Datasets
Download the KDD-1999 dataset at https://datahub.io/machine-learning/kddcup99 and the CIC-2017 dataset on Kaggle at https://www.kaggle.com/datasets/cicdataset/cicids2017/code.

# Mechanics
See the uploaded paper or https://ia601602.us.archive.org/25/items/pioneer-su-21-sophia-zhu-term-paper/Pioneer_Su21_Sophia_Zhu_Term_Paper.pdf for reference.

# Notes
This project is part of an individual research during the Pioneer summer program. The model is robust on both datasets. Since the datasets provide different columns, two separate systems are trained based on KDD-99 and CIC-17.
