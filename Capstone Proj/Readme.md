# **MUSIC: Make Unification Simple in Image Classification**

## **Overview**
This project demonstrates the concept of **Task Vectors**, focusing on fine-tuning pre-trained models for specific tasks and analyzing the resulting performance improvements. The experiments are conducted on the MNIST dataset.

---

## **Key Steps**

- **Dataset Preparation**: MNIST dataset is used, with specific classes (3, 5, and 6) selected for task-specific fine-tuning.
- **Model Fine-Tuning**: Fine-tuned a linear model for individual classes.
- **Task Vectors**: Computed task vectors as the difference in model weights before and after fine-tuning.
- **Evaluation**: Tested the models using accuracy, confusion matrices, and other metrics.

---

## **Dataset**

- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- **Classes Used**: 3, 5, and 6  
- **Training Size**: 1000 samples per class  

---

## **Technologies Used**

- **Language**: Python  
- **Libraries**:
  - PyTorch  
  - NumPy  
  - Matplotlib  
  - Pickle  

---

## **Workflow**

1. Load the MNIST dataset and preprocess it.  
2. Fine-tune a base model for classes 3, 5, and 6.  
3. Compute task vectors based on the fine-tuning results.  
4. Apply task vectors to assess transfer learning.  
5. Evaluate performance using accuracy and confusion matrices.  

---

## **Results**

- **Base Model Performance**: Achieved ~85-90% accuracy on the MNIST dataset.  
- **Task-Specific Fine-Tuning**: Improved performance on classes 3, 5, and 6.  
- **Task Vector Transfer**: Demonstrated effective knowledge transfer.  

---

## **Future Improvements**

- Extend experiments to more diverse datasets.  
- Explore task vectors in different domains like text or audio.  
- Optimize the workflow for deployment on edge devices.  

---
