# Dino Player Reinforcement Learning Model

This model is built using [TensorFlow.js](https://www.tensorflow.org/js) Node.js implementation and it's using the DDQN algorithm to play the Dino Chrome game. Browser automation is done using [Puppeteer](https://pptr.dev/).

## Steps

1. Clone the repository:
git clone https://github.com/gajojr/tfjs-dino-player.git

2. Change to the project directory:
cd tfjs-dino-player

3. Install the dependencies:
npm install

4. Clone the Dino Chrome game repository:
git clone https://github.com/wayou/t-rex-runner.git

5. Install `http-server` globally:
npm i -g http-server

6. Change to the `t-rex-runner` directory:
cd t-rex-runner

7. Start the `http-server` in the `t-rex-runner` directory:
http-server

8. Open another command line in the `tfjs-dino-player` directory.

9. Run the Dino bot with the desired mode:
- To train the model, run:
  ```
  node dino-bot.js --train
  ```
- To play the game, run:
  ```
  node dino-bot.js --play
  ```
