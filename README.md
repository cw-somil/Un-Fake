# Un-Fake
Fake News Detector using Deep NLP and OCR 
## Winning project for the Enroot Innovate for Mumbai Hackathon

# Problem Statement
To create a product for detecting Fake News in Mumbai

# Description

Created a Website as well as a Google Assistant Action for users to easily predict Fake News. Developed a Deep Neural Network using Bidirectional LSTMs to train on 7000 articles labelled as Real/Fake. Gained 90% accuracy after various advanced Text Preprocessing.
Added functionality of OCR using OpenCV and Tesseract OCR to scan any news article and predict using the Deep NLP model. 

# Website

The Website is developed on Django and we have currently implemented two functionalities,
1) Text - Directly input an article and test the Reliability of the News.
2) Image - Input an article image and predict the reliability through OCR and Deep NLP.

The How page shows how much data we have currently predicted has shown has Reliable or Not.


# Google Action

Developed a Google Assistant Action named "News Companion" to help users detect fake news easily. Used the Actions SDK to connect the Google Action to our model and provide the required output.


# Future Scope
We are working on providing better functionalities and improved accuracy to make it as perfect as possible.
Possible functionalities to be added in the future
1) Google Chrome Extension to directly access the detector on any page
2) Twitter and Facebook integration to verify reliability of articles
3) Whatsapp integration to prevent spreading of Fake articles
