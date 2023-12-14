# DistilIQA: Distilling Vision Transformers for no-reference perceptual CT Image Quality Assessment

No-reference image quality assessment (IQA) is a critical step in medical image analysis, with the objective of predicting perceptual image quality without the necessity for a pristine reference image. The application of no-reference IQA to CT scans is valuable to provide an automated and objective approach to assessing scan quality, optimizing radiation dosage, and enhancing overall healthcare efficiency. In this paper, we introduce DistilIQA, a novel distilled Vision Transformer network designed for no-reference CT image quality assessment. DistilIQA integrates convolutional operations and multi-head self-attention mechanisms by incorporating a powerful convolutional stem at the beginning of the traditional ViT network. Furthermore, we present a two-step  distillation methodology (presented in the Figure below) aimed at improving network performance and efficiency. In the initial step, a "teacher ensemble network" is constructed by training five vision Transformer networks through a five-fold division schema. In the second step, a "student network", comprising of a single Vision Transformer, is trained using the original labeled dataset and the predictions generated by the teacher network as new labels. DistilIQA is evaluated in the task of quality score prediction from low-dose chest CT scans obtained from the LDCT and Projection data of the Cancer Imaging Archive, along with low-dose abdominal CT images from the LDCTIQAC2023 Grand Challenge. Our results demonstrate DistilIQA`s remarkable performance in both bechmarks, surpassing the capabilities of various CNNs and Transformer architectures. Moreover, our comprehensive experimental analysis demonstrates the effectiveness of incorporating convolutional operations within the ViT architecture and highlights the advantages of our distillation methodology.  
![alt text](https://github.com/mariabaldeon/DistilIQA/blob/main/Images/Framework.jpg)
