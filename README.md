# CodeIt with Improved data augmentation
This repository extends CodeIt by enhancing its data augmentation process through two key mutation improvements and their combinations. These improvements aim to increase task diversity while mitigating redundancy, ultimately improving the model's generalization. The frame work of [CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://arxiv.org/abs/2402.04858)
is published at ICML 2024. The usage of this repository remains largely the same as the original CodeIt implementation. Please refer to the original documentation for setup and usage instructions [https://github.com/Qualcomm-AI-research/codeit/blob/main/README.md].



## Abstract

> This study investigates the potential of neurosymbolic approaches to solve the Abstraction and Reasoning Corpus (ARC) challenge focusing on the CodeIt framework. Improvements were made to CodeIt by augmenting its dataset to enhance generalization and incorporating a model fine-tuned on reverse-engineered DSL training tasks. Data augmentation increased task diversity but did not consistently improve performance, likely due to the inclusion of lower-quality tasks that were difficult to filter out. Using a DSL fine-tuned model demonstrated improved performance compared to a baseline model, highlighting the potential benefits of pretraining using DSL tasks in the CodeIt framework.
