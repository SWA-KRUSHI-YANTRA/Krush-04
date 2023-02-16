import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the data generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'cotton_images',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

# Load the Inception model and add a custom head
base_model = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)

# Save the trained model
model.save('cotton_detection_model.h5')
