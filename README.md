# Fullscale-AI-Project

Image Trainer
```py
from imageTrainer import imageTrainer

imagetrainer = imageTrainer('./train_data')

imagetrainer.train()
```

this will create and save a tf model that classifies an image

File Structure:

```
├── images                  # images folder named anything
│   ├── train               # train folder, named anything, make sure this is the folder you pass into imageTrainer
│       ├── class #1        # named the first image type, containing all of that type, Ex. Waving, or Clapping
│       ├── class #2        # named second class, whatever that may be
```

you can have as many classes as you want

Classification of data:

```py
classification = imagetrainer.classify('image.png)
```

using an image trainer which you have defined, this will return the classification of a given image, if the image is not in the dataset you WILL get wrong, or strange outputs

Parameter:
```py
 data_dir, validation_split = 0.2, batch_size = 32, image_sizes = 180, optimizer = 'adam', epochs = 15, export = './trainedModel'
 ```
 
 data_dir is the required parameter into imageTrainer, everything else can be edited, but should be quite optimal
 
 validation split - what percent of data is used for validation
 
 batch size - batch size the nn (neural network) uses
 
 image sizes - what the nn scales images down to, and processes, important to make sure it isnt overloaded with data
 
 optimzer - TensorFlow optimzer, list of optimizers can be found on their website, but adam should be very good for most circumstances
 
 eopochs - training length, 15 works just fine for semi accurate data, for extremly accurate data though, if you have a lot of epochs you risk overtraining
 
 export - export file location of saved model, this can be used by importing it into a tensorflow neural network any time later
 
 you can import a saved model with
 
 ```py
 model = tf.keras.models.load_model('./trainedModel')
 ```
 

