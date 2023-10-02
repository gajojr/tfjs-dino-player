# Play Mode Description

This document outlines the steps followed in the `setupModelPlaying` function to play the game using a trained model.

## Execution Steps

1. Call the `setupModelPlaying` asynchronous function to initiate the play mode.
2. Create a `gameProxy` object by awaiting the `createChromeGameProxy` function.
3. Instantiate a `DinoAgent` object using the `gameProxy` object.
4. Load the existing trained model:
    - Try to load the saved target model using `tf.loadLayersModel` from the specified path.
    - Compile the loaded model using the `compileModel` function.
    - Log a success message to console upon successful model loading.
    - Set `dinoAgent.epsilon` to `0` to ensure the model always chooses the best action.
    - If no model is found, log an error message to the console and exit the process.
5. Create an online model instance using `dinoAgent.createModel`, and set its weights to match those of the target model. Then compile the online model using the `compileModel` function.
6. Ensure a fresh game start by:
    - Awaiting `gameProxy.restart` to restart the game.
    - Delaying for a brief period using `dinoAgent.delay` to allow the game to settle.
    - Triggering the start of the game by executing `gameProxy.jump`.
7. Obtain the initial game state by awaiting `gameProxy.state`.
8. Enter an infinite loop to keep playing the game:
    - Convert the current game state to a vector using `dinoAgent.stateToVector`.
    - Select an action using `dinoAgent.selectAction`.
    - Perform the selected action using `gameProxy.performAction`.
    - Await the next game state using `gameProxy.state`, and continue to do so until a significant change in game time is observed or the game ends.
    - Update the current state to the next state for the next loop iteration.
    - If the game ends (`state.done` is `true`), restart the game and continue playing from step 6.

## Note

- The `setupModelPlaying` function allows the trained model to play the game in an automated fashion, choosing the best action at each step based on the game state.
- The infinite loop ensures continuous gameplay, with the game restarting automatically whenever the agent crashes.