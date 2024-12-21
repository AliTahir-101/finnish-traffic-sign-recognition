# 🚦 Finnish Traffic Sign Recognition with Synthetic Data 🚘

Welcome to the repository for **"Enhancing Real-World Finnish Traffic Sign Detection and Classification Using Synthetic Data Augmentation and Transfer Learning Techniques"**. This project is part of my Master’s thesis in Artificial Intelligence at JAMK University of Applied Sciences.

---

## 📖 About the Project

This project explores the use of synthetic data, transfer learning, and advanced models to improve the detection and classification of Finnish traffic signs. The work focuses on:
- Generating synthetic datasets with 38 and 55 traffic sign classes using Blender and OpenCV.
- Training state-of-the-art classifiers like `EfficientNetB0` for improved accuracy.
- Developing a real-time detection pipeline using `YOLOv11`.
- Comparing `YOLOv11` with a hybrid approach: `YOLOv11 + EfficientNetB0`.

---

## 📊 Datasets

### 1. 🖼️ Cropped Traffic Sign Dataset
- [**38 Classes**](https://drive.google.com/file/d/1YFE1j81nrWktzH0hisaKpJc9ig6UOZqT/view?usp=sharing): A synthetic dataset of cropped traffic sign images representing 38 Finnish traffic signs.
- [**55 Classes**](https://drive.google.com/file/d/1yc9oF-qecFO9CkpS3YrlC7WyCq4XU8Ml/view?usp=sharing): An extended dataset including additional traffic sign types for a total of 55 classes.
- [**55 Classes with Transparent Backgrounds**](https://drive.google.com/file/d/1EgoCKEISKd-_xttAX4wILchWsajzQeMz/view?usp=sharing): A synthetic dataset with transparent backgrounds to facilitate further research and synthetic data creation. These images can be directly used for data augmentation or inserted into real-world scenes using tools like OpenCV or Blender.
### 2. 📏 YOLO Dataset
- Images with bounding boxes for real-time traffic sign detection.

---

## 🖼️ Synthetic Dataset Previews
<video src='https://github-production-user-asset-6210df.s3.amazonaws.com/76158157/397943738-d8c9acb4-2a82-4aa1-aea5-16f25a07a178.mp4' width=640/></video>

### 1️⃣ Cropped Images for Classifiers
Below is a preview of the **synthetic dataset of cropped images** used to train classification models:

![Cropped Dataset Preview](https://github.com/user-attachments/assets/d95b54f2-ca87-4e82-8e8b-d66e85140efd)

These images represent traffic signs with transparent backgrounds generated from SVG files, augmented for diverse angles and variations.

---

### 2️⃣ YOLO Dataset with Bounding Boxes
Here’s a sample collage from the **YOLO dataset** showcasing labeled bounding boxes for traffic signs:

![YOLO Dataset Preview](https://github.com/user-attachments/assets/3f38a2ac-d2e7-46d7-9a3e-bd20630a08dc)

This dataset includes realistic scenes with traffic signs, annotated with bounding boxes for object detection.

---

## 🔄 How to Create a Synthetic Dataset

If you're interested in creating a synthetic dataset from SVG files, I’ve provided a guide in a separate documentation file. This guide covers the process of generating a transparent background dataset using Blender and augmenting it with realistic backgrounds for machine learning model training.

👉 **[Read the full guide here](./how_to_create_synthetic_dataset.md)**

This guide includes:
- Setting up Blender and running the provided Python script for SVG processing.
- Using a Python script to create a final augmented dataset with real-world backgrounds.
- Tips and notes to customize the scripts and optimize your dataset creation process.


---

## 🖼️ Dataset Visualization

Below is a collage of the **55 Finnish Traffic Sign Classes** used to create the dataset and train the models. These include the original 38 classes and the newly added 17 classes for a total of 55:

![Collage of 55 Finnish Traffic Sign Classes](https://github.com/user-attachments/assets/ee67cf75-1b3e-40f2-ac84-761647601655)

### 🔗 Original SVG Files

The original SVG files for Finnish traffic signs were sourced from the [Finnish Traffic Agency GitHub Repository](https://github.com/finnishtransportagency/liikennemerkit/tree/master/collections/new_signs/svg). These files were processed using **Blender** and **OpenCV** to generate synthetic datasets with realistic augmentations, including lighting, motion blur, and various environmental conditions.

---

## 🖼️ Classification Results Sample

Below is a sample collage of classification results for the **55-class model** on real-world data:

![Classification Results Collage](https://github.com/user-attachments/assets/aa89ec23-1adc-4db2-834c-85e483baf27a)


These samples demonstrate the model's ability to accurately classify Finnish traffic signs under various environmental conditions, including different lighting, perspectives, and occlusions.


---

## 🧑‍💻 Notebooks

1. [**`01_finnish_traffic_sign_classifier_38_classes_jhoonas_dataset_99_36_cnn.ipynb`**](./notebooks/01_finnish_traffic_sign_classifier_38_classes_jhoonas_dataset_99_36_cnn.ipynb): Covers preprocessing, training, and testing using the 38-class dataset and a custom CNN model.
2. [**`02_finnish_traffic_sign_classifier_my_custom_dataset_38_classes_99_95_EfficientNetB0.ipynb`**](./notebooks/02_finnish_traffic_sign_classifier_my_custom_dataset_38_classes_99_95_EfficientNetB0.ipynb): Includes preprocessing, training, and testing of the EfficientNetB0 model on the enhanced 38-class dataset.
3. [**`03_transfer_learning_finnish_traffic_sign_classifier_55_classes_99_100_EfficientNetB0.ipynb`**](./notebooks/03_transfer_learning_finnish_traffic_sign_classifier_55_classes_99_100_EfficientNetB0.ipynb): Details preprocessing, transfer learning, training, and testing on the 55-class dataset using EfficientNetB0.

---

## 🎥 Demo Video
Check out the YOLOv11 vs YOLOv11 + EfficientNetB0 comparison video to see the models in action! 🚦📹

<video src='https://github-production-user-asset-6210df.s3.amazonaws.com/76158157/395735455-3cb21a0c-3d02-41fe-a075-0853dedd4f8b.mp4' width=640/></video>


---

## 📊 Results

| **Metric**                     | **Custom CNN (38 Classes – Old Dataset)** | **EfficientNetB0 (38 Classes – New Dataset)** | **EfficientNetB0 (55 Classes – New Dataset)** | **YOLOv11 (Bounding Box Detection)** |
|--------------------------------|-------------------------------------------|-----------------------------------------------|-----------------------------------------------|----------------------------------------|
| **Dataset Images/Class**       | 720                                       | 2276                                          | 2276                                          | 2276                                   |
| **Training**                   |                                           |                                               |                                               |                                        |
| Epochs                         | 35                                        | 15                                            | 25                                            | 30                                     |
| Accuracy                       | 99.81%                                    | 99.79%                                        | 99.21%                                        | Final Training Box Loss: 0.2324       |
| Loss                           | 0.1403                                    | 0.0064                                        | 0.0148                                        | Final Training Class Loss: 0.2354     |
|                               |                                           |                                               |                                               | Final Training DFL Loss: 0.7698       |
| **Validation**                 |                                           |                                               |                                               |                                        |
| Accuracy                       | 99.87%                                    | 99.94%                                        | 98.71%                                        | Final Validation Precision: 0.9515    |
| Loss                           | 0.0122                                    | 0.0018                                        | 0.0294                                        | Final Validation Recall: 0.9975       |
| F1 Score                       | 1.00                                      | 1.00                                          | 0.99                                          | Final Validation mAP@50: 0.9745       |
| Precision                      | 1.00                                      | 1.00                                          | 0.99                                          | Final Validation mAP@50-95: 0.9587    |
| Recall                         | 1.00                                      | 1.00                                          | 0.99                                          | Validation Box Loss: 0.2071           |
|                               |                                           |                                               |                                               | Validation Class Loss: 0.2408         |
|                               |                                           |                                               |                                               | Validation DFL Loss: 0.7611           |
| **Testing (Synthetic)**        |                                           |                                               |                                               |                                        |
| Accuracy                       | 99.82%                                    | 99.90%                                        | 98.71%                                        |                                        |
| **Testing (Real-World)**       |                                           |                                               |                                               |                                        |
| Accuracy                       | 35.90%                                    | 94.87%                                        | 100.00%                                       |                                        |
| Accuracy (All 55 Classes)      | ---                                       | ---                                           | 96.30%                                        |                                        |
| **Observations**               | Limited generalization capability         | Significant improvement over Custom CNN, achieving excellent generalization capabilities. | Transfer learning significantly improved performance across all datasets. | The YOLOv11 model demonstrates robust performance with high precision, recall, and mAP values, indicating effective bounding box detection under various conditions. |
| **Room for Growth**            | ---                                       | - EfficientNetB0 could be improved with images of varying resolutions to better handle small and large signs. Current training on 100x100 images limits generalization to different sizes. | - EfficientNetB0 performance could be enhanced by introducing more diverse augmentations and additional variations in real-world scenarios. | - YOLOv11 could be trained further to improve generalization. Expanding training datasets with more diversity and fine-tuning hyperparameters could yield better results. Resource and time constraints limited these optimizations. |

---



## ✨ Acknowledgements

Special thanks to:

- [**Tomi Nieminen**](https://www.jamk.fi/en/expert/tomi-nieminen), my supervisor from the JAMK Department of Teknologia, School of Technology, for his guidance and expertise.
- [**Mika Rantonen**](https://www.jamk.fi/en/expert/mika-rantonen), my coordinator from the Department of IT-Instituutti, Institute of Information Technology, for his support throughout this project.
- [**Joona Tenhunen**](https://urn.fi/URN:NBN:fi:amk-2021052812336), the previous researcher, whose 38-class dataset was instrumental in creating the baseline model and initiating this work.
- [**JAMK University of Applied Sciences**](https://www.jamk.fi/en) for providing the resources and platform to conduct this research.
- **Finnish Traffic Agency** for providing [SVG resources](https://github.com/finnishtransportagency/liikennemerkit/tree/master/collections/new_signs/svg).
- Open-source contributors of **Blender**, **OpenCV**, and **PyTorch** for their invaluable tools and frameworks.


---

## 📬 Contact

Feel free to reach out with any questions or suggestions!  

📧 **contact@alitahir.dev**  
🌐 **[alitahir.dev](https://alitahir.dev)**  
