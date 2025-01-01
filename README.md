# Image Captioning with PyTorch  

## 🌟 Project Overview  
This project explores the exciting field of **image captioning**, where computer vision meets natural language processing to generate meaningful captions for images. With a backbone of **ResNet34** and the extensive **COCO dataset**, the project aims to create a robust model capable of understanding images and describing them in natural language.  

## 🔍 Real-World Applications  
- **Accessibility**: Generating descriptive captions for visually impaired users.  
- **E-commerce**: Automating product tagging and description creation.  
- **Content Creation**: Enhancing social media workflows with automatic captions.  
- **Media Management**: Improving image search and categorization.  
- **Healthcare**: Assisting in medical imaging annotation.  

##  Project Highlights  
- **Model Architecture**: Utilized **ResNet34** as a feature extractor for its efficiency and performance balance.  
- **Dataset**: Trained on the **COCO dataset**, containing more than 200,000 lebeled images paired with five captions each.  
- **Framework**: Developed using **PyTorch**, chosen for its intuitive object-oriented programming style and flexibility.  

## 🛠️ Model Architecture

### **1. CNN Encoder: ResNet-34**
The encoder of the image captioning model is based on a pretrained **ResNet-34** architecture, a deep convolutional neural network known for its high representational power. Here’s how the encoder works:

- **Feature Extraction**:
  - The top fully connected (fc) layer of ResNet-34 is removed, leaving the convolutional layers intact.
  - The resulting architecture outputs feature vectors that represent the high-level features of the input images.
- **Custom Layers**:
  - A fully connected (fc) layer maps the extracted features to a lower-dimensional space of size `embedding_size`.
  - Batch normalization is applied to stabilize and speed up training.


### **2. LSTM Decoder**
The LSTM layer takes as input Image embeddings + Word embeddings and outputs Word probability distributions (captions)

Caption Generation:
Uses greedy search to predict captions during inference.


👉 **Try it out here**: [Streamlit App](https://image-captioning-aka5tny4hjsehrjzv6vhms.streamlit.app)


## ⚙️ Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Abdulmateen7827/Image-Captioning.git 
   cd Image-captioning  

2. Install dependencies
```python 
pip install -r requirements.txt
```
3. Run the application
```python
streamlit run streamlit_app.py
```
## Future Enhancements  
- **Leverage Deeper Architectures** 
- **Incorporate Attention Mechanisms**
- **Expand to Video Captioning**
- **Diversify Captions** 
- **Enable Multi-Language Support**
- **Adopt Beam Search for more accurate captions**
