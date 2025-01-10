 How to download image model in google teachable machine:

 Step1:Search in web browser google teachable machine
 ![Screenshot 2025-01-09 234718](https://github.com/user-attachments/assets/2d95b848-1466-48fe-b643-26b23fbec4cb)

 Step2: click on Teachable Machines website:
 ![Screenshot 2025-01-09 234841](https://github.com/user-attachments/assets/927c4476-7ecc-475b-9ec5-f7a5ec8b4c5a)

 Step3: Click on Get Started:
 ![Screenshot 2025-01-09 234913](https://github.com/user-attachments/assets/48b52151-8610-4288-8fed-a40b9d3ccdcc)

 Step4: Click on Image Project:
 
 ![Screenshot 2025-01-09 234950](https://github.com/user-attachments/assets/3fe6ffbe-8fd8-450a-8b7d-6231d892607f)

 Step5: Click on Image model:
 ![Screenshot 2025-01-09 235016](https://github.com/user-attachments/assets/85ae1e03-1f6a-447c-97ff-5906e3d776fc)

 Step6:Train your model in classess and add data samples to it :
 ![Screenshot 2025-01-09 235409](https://github.com/user-attachments/assets/8e35db10-a1a0-46dc-8ce5-0885bb30e120)

 Step7: Train your model according to your classes:
 ![Screenshot 2025-01-09 235455](https://github.com/user-attachments/assets/86e0e7f7-8cbc-45a7-b70e-e9b7dd8ff3d9)

 Step8: Export the trained model you will get the code in tensorflow. Open (opencv keras code). Click on Download model
 ![Screenshot 2025-01-09 235553](https://github.com/user-attachments/assets/62bb4860-c458-4eac-9831-4a1459303de4)

 Step9: place the downloaded model in your project directory, such as(keras h5.model and labels.txt)
 
![image](https://github.com/user-attachments/assets/4246807a-a477-4c8a-aece-66c487763b8a)

Step10: create a python file in your pycharm IDE as (main.py)


from tensorflow.keras.models import load_model
# TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels, stripping any newline characters
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)




while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name, end="")  # Removed slicing to print the full class name
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

 Step11: Run the code output will be expected with the confidence score of your model in pycharm. It will directly connect to Your Webcam directly and give you the output

 Step12:Expected errors 

 Installation of neccesity libraries and missing packages
 Update the tensorflow according to the version of Your vitrual environment

  Note: While creating the new project in pycharm create a proper virtualenvironment and version of python.
 




 





