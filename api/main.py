from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image  #Pillow is module used tpo read images in a python
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Lets create as global variable named MODEL
MODEL = tf.keras.models.load_model("/Users/umakantmukhiya/Code/potato-disease/saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  #class_name must be consistent with what we had in ipynb notebook while traning


@app.get("/ping")  #end point
async def ping():
    return "Hello I'am alive"

def read_file_as_image(data) -> np.ndarray:
    #here data variable is bytes, hence you can use BytesIO python module 
    image = np.array(Image.open(BytesIO(data)))  #it will read these bytes as an Pillo w Image
    return image

@app.post("/predict")  #end point
async def predict(
    # here we will have file(image of leaf)  sent by mobile or web application to our model
    #if we use upload file as a data type
    #since fastAPI uses inbuilt validation
    file: UploadFile = File()

):
    image = read_file_as_image(await file.read()) #why we are using async and await here? find on youtube
    #Now that we got the file from user, jow to convert iinto numpy() array 
    #above image variable is our numpy() array

    img_batch = np.expand_dims(image,0)  #read about np.expand_dims on google

    prediction = MODEL.predict(img_batch)  #this predict function doesn't take single image as an input, it takes multiple images
    predicted_class = CLASS_NAMES[ np.argmax(prediction[0]) ] #since its a batch , but we have given only single image hence 0 index
    # np.argmax(prediction[0])  return index
    confidance = np.max(prediction[0])  #  np.max(prediction[0]) will return max value in list

    return {
        'class': predicted_class,
        'confidence': float(confidance)
    }

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8000)