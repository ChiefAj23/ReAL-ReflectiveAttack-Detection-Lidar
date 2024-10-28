
# ReAL: Machine Learning Detection of Reflective Attacks against Lidarometry

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-green.svg)](https://www.python.org/)

**Authors**: Abhijeet Solanki<sup>1</sup>, Luke Beirne<sup>2</sup>, Syed Rafay Hasan<sup>3</sup>, Wesam Alamiri<sup>3</sup>  
<sup>1</sup>Department of Electrical and Computer Engineering, Tennessee Technological University  
<sup>2</sup>Department of Computing Sciences, Coastal Carolina University

---

## Overview
This repository contains code and resources for the project **ReAL**, focusing on detecting reflective surface interference in LiDAR readings. Our approach employs a machine learning-based system for real-time detection of reflective interference on resource-constrained devices.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [References](#references)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ChiefAj23/ReAL-ReflectiveAttack-Detection-Lidar.git
cd ReAL-ReflectiveAttack-Detection-Lidar
```
```bash
# Install dependencies
pip install -r requirements.txt
```
## Requirements
Python 3.7+
Jetson Orion Nano (for resource-constrained testing)
LiDAR sensor (RPLiDAR A1M8-R6 recommended)
Make sure to set up the hardware and sensor according to the manufacturer’s guidelines.

Usage
To begin detecting reflective attacks on LiDAR data:

Set up the LiDAR sensor and prepare your testing environment, ensuring all necessary safety protocols are followed.
Run the detection command with preprocessed LiDAR data:
```bash
python detect_reflective_attack.py --input data/lidar_data.csv --model checkpoints/svm_model.pkl
```
This command initiates real-time interference detection using the specified model and input dataset.

## Dataset
For instructions on preparing and structuring data, see the data/README.md.

## Experiments
We conducted controlled experiments to assess the accuracy and response time on reflective surface detection. The repository includes configurations for replication or further experimentation.

See experiments/README.md for setup details.

## References
Relevant references are included in the paper.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


