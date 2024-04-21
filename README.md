# Gesture Vision

<h1> Dependencies</h1>
There will be requirement of installing dependencies based on the modules used in this project like tensorflow, keras, fast autocomplete[leverkusen], gTTs and etc. <br><br>
<h1> How to run the project</h1>
To recreate the 2D CNN model that has been used for this tool:<br>
1. First get in to the directory of 2d_cnn <br>
2. Run the create_dataset.py file. Ensure that the data direrctory is correct. If this is executed successfully, a pickle file of the data will be created. <br>
3. Next step is Training the 2D CNN. Run the train_2D_cnn.py file. Ensure that the data directory is correct. This will train and save the model to the cnn_model_2d.h5. <br>
4. Now train the RNN model for next word prediction by running next_word_predictor.py file. This will save next_word_model.h5 file. <br>
5. Use this model to detect characters in realtime by running the image_classifier_cnn.py.<br>

<br>For prediction of the character, signs to be used can be found in asl_dataset folder in main folder. Inside asl_dataset, folders 0 to 25 contains sign for A to Z, folder 26 is for taking next frame, 27 is for space, 28 is for taking current word suggestion, 29 is for taking next word suggestion, 30 is for backspacing in current word and 31 is for converting text to speech. After each character predicted user need to provide next sign, sign in folder 26, inorder to give next sign as input. If this is not provided then user won't be able to provide any sign. For example, If sign A is detected then user needs to provide next sign gesture and only then B sign gesture can be given. This feature helps to give accurate input to model.

<br>Details can be found in Project Report.
