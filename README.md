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
- **38 Classes**: A synthetic dataset of cropped traffic sign images representing 38 Finnish traffic signs.
- **55 Classes**: An extended dataset including additional traffic sign types for a total of 55 classes.

### 2. 📏 YOLO Dataset
- Images with bounding boxes for real-time traffic sign detection.

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

## 📈 Results

To be shared.

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
