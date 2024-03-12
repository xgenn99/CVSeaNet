# CVSeaNet
Undesirable activities like the illegal dumping of pollutants, unauthorized fishing, human and goods trafficking and piracy require great attention from institutions. Although the Automatic Identification System continues to play a crucial role in current Vessel Monitoring Systems, transponders can be intentionally turned off, and their usage is confined to specific vessel classes and dimensions. SAR imagery offers diverse data types and imaging capabilities and its integration with maritime surveillance ensures cost-effective, global coverage, and a continuous stream of data for ship detection. Traditional algorithms, like CFAR, were prevalent in this domain, but with the emergence of Deep Learning (DL), they have been eclipsed in terms of both speed and precision. Neural Networks significantly boost the overall detection system, showcasing remarkable resilience and flexibility across diverse, intricate scenarios, thereby eliminating the requirement for segmenting between sea and land in nearshore and port environment. Current DL approaches for ship detection primarily operate on SAR image intensity rather than harnessing the complete potential of pixel complex values present in the Single Look Complex (SLC) data. Thus, the primary research objective of this work is the development of a novel ship detection Neural Network, called CVSeaNet (Complex-Valued Sea Net), which handles SAR images with complex weights and biases and not as pure intensity products. CVSeaNet is trained with a custom multi-frequency SAR SLC dataset enriched with additional information such as multiple polarizations, incidence angle, and AIS information, called Multi-frequency SAR Single Look Complex Ships (MS3). The baseline architecture of CVSeaNet comprises a feature extraction backbone and a prediction head. Developed in Python, through PyTorch framework and using VSCode, the backbone is inspired by EfficientNet B0 model, but with complex-valued layers. The head, instead, is similar to CenterNet, an anchor-free detector that scales down output feature maps to save memory. CVSeaNet is trained with images from MS3 dataset on a NVIDIA GeForce 3060 GPU (12 GB) with a batch size of 3, utilizing a learning rate of 1e-2. Two cases are considered for results validation: CVSeaNet and the same architecture with real-valued convolution instead of complex layers. A third case is evaluated considering the same complex-valued architecture but with the late data fusion of the incidence angle. Even if the third case exhibits, as expected, an improvement in the evaluation metrics, it is not considered because it does not show significant differences compared to the complex-valued in terms of outcomes. The experimental results, conducted on CVSeaNet using the MS3 dataset, illustrate
the effectiveness of the complex-valued method compared to the case in which real-valued convolution is employed considering the same architecture, showing an increase of both precision and recall (∼ 2% and ∼ 3% respectively). Finally, it is reasonable to affirm that the study provides a robust baseline for future research
endeavors in object detection applied to maritime surveillance.
