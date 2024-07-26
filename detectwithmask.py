import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
video = cv2.VideoCapture(0)
while True:
    ret,frame = video.read()
    img = cv2.resize(frame,(224,224))
    test_image = np.array(img,dtype=np.float32)
    test_image = np.expand_dims(test_image,axis=0)
    normalizeimage = test_image/255.0
    prediction = model.predict(normalizeimage)
    print("predicción",prediction)
    cv2.imshow("fotograma",frame)
    key=cv2.waitKey(1)
    if key == 32:
        break
video.release()

