# FSMLP: Frequency Simplex MLP for Time Series Forecasting

![FSMLP Method Overview](method.pdf)

## Introduction

Time series forecasting (TSF) is a crucial task in various domains such as web data analysis, energy consumption prediction, and weather forecasting. Traditional Multi-Layer Perceptrons (MLPs) are lightweight and effective in capturing temporal dependencies but often suffer from overfitting when modeling inter-channel dependencies. 

To address this issue, we introduce a novel Simplex-MLP layer inspired by simplex theory. By constraining the weights within a well-defined standard simplex, the Simplex-MLP enforces the model to learn simpler patterns, thus reducing the impact of extreme values and effectively capturing inter-channel dependencies.

Based on the Simplex-MLP layer, we propose the Frequency Simplex MLP (FSMLP) framework for time series forecasting. The FSMLP framework consists of:
- **Simplex Channel-Wise MLP (SCWM)**: Utilizes Simplex-MLP to capture inter-channel dependencies.
- **Frequency Temporal MLP (FTM)**: A simple yet effective temporal MLP that extracts temporal information.

The FSMLP framework operates in the frequency domain rather than the time domain. Each frequency component corresponds to periodic patterns in the time domain, allowing FSMLP to capture relationships between periodic patterns across channels, which helps mitigate noise and reduce overfitting.

## Theoretical Analysis

Our theoretical analysis shows that the upper bound of the Rademacher Complexity for Simplex-MLP is lower than that for standard MLPs, indicating improved generalization capabilities.

## Experimental Validation

We validate our method on seven benchmark datasets, demonstrating significant improvements in forecasting accuracy, efficiency, and scalability. Additionally, we show that Simplex-MLP can improve other methods that use channel-wise MLPs, leading to reduced overfitting and better performance across a range of forecasting tasks.

## Installation

To install and use FSMLP, follow these steps:

```bash
git clone https://github.com/yourusername/FSMLP.git
cd FSMLP
pip install -r requirements.txt
