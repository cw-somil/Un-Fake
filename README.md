# Un-Fake
Fake News Detector using Deep NLP and OCR 
## Winning project for the Enroot Innovate for Mumbai Hackathon


### Problem Statement
To create a product for detecting Fake News in Mumbai

### Description

Created a Website as well as a Google Assistant Action for users to easily predict Fake News. Developed a Deep Neural Network using Bidirectional LSTMs to train on 7000 articles labelled as Real/Fake. Gained 90% accuracy after various advanced Text Preprocessing.
Added functionality of OCR using OpenCV and Tesseract OCR to scan any news article and predict using the Deep NLP model. 

### Model Architecture used

Used Many to one Bidirectional LSTMs to get the prediction.

![Many to one architecture](https://github.com/Somil112/Un-Fake/blob/master/Screenshots/ss3.jpg)

### Website

The Website is developed on Django and we have currently implemented two functionalities,
1) Text - Directly input an article and test the Reliability of the News.
2) Image - Input an article image and predict the reliability through OCR and Deep NLP.

The How page shows how much data we have currently predicted has shown has Reliable or Not.

<img src="https://github.com/Somil112/Un-Fake/blob/master/Screenshots/ss1.png" height="70%" width="70%" >
<img src="https://github.com/Somil112/Un-Fake/blob/master/Screenshots/ss4.png" height="70%" width="70%" >
<img src="https://github.com/Somil112/Un-Fake/blob/master/Screenshots/ss2.png" height="70%" width="70%" >


### Google Action

Developed a Google Assistant Action named "News Companion" to help users detect fake news easily. Used the Actions SDK to connect the Google Action to our model and provide the required output.

<img src="https://github.com/Somil112/Un-Fake/blob/master/Screenshots/ss5.png" height="40%" width="40%" >

### Data used

Stanford pretrained Glove Vectors for Word Embeddings <br>
7000 articles labelled fake or real.

### Future Scope
We are working on providing better functionalities and improved accuracy to make it as perfect as possible.
Possible functionalities to be added in the future

1) Google Chrome Extension to directly access the detector on any page
2) Twitter and Facebook integration to verify reliability of articles
3) Whatsapp integration to prevent spreading of Fake articles
