# Training Mode Description

This document outlines the steps encapsulated in the `setupModelTraining` asynchronous function to set up and train a model for playing the game.

## Execution Steps

1. Call the `setupModelTraining` asynchronous function to initiate the training mode.
2. Create a `gameProxy` object by awaiting the `createChromeGameProxy` function.
3. Instantiate a `DinoAgent` object using the `gameProxy` object.
4. Load the existing trained model or create a new one:
    - Try to load the saved target model using `tf.loadLayersModel` from the specified path.
    - Compile the loaded model using the `compileModel` function.
    - Log a success message to the console upon successful model loading.
    - Set `dinoAgent.epsilon` to `0` to ensure the model behaves greedily. This is crucial to see the effect of a loaded model, as an epsilon value of `1` would cause rapid failure.
    - If no model is found, log a message to the console indicating the creation of a new model, and then create a new target model using `dinoAgent.createModel`.
5. Create an online model instance using `dinoAgent.createModel`, and set its weights to match those of the target model. Then compile the online model using the `compileModel` function.
6. Specify the number of training episodes as `100000`.
7. Initiate the training process by awaiting `dinoAgent.trainModel` with the specified number of episodes.

## Note

- The `setupModelTraining` function either loads an existing model or creates a new one for training purposes.
- Two models are utilized: an online model for ongoing training and a target model to stabilize the Q-value targets.
- Training is conducted over a specified number of episodes to improve the model's performance over time.
