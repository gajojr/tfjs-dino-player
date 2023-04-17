const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');
const readline = require('readline');
const colors = require('colors');
const fs = require('fs').promises;
const Memory = require('../src/Memory');
const { computeReward, selectAction, performAction, proxy } = require('../src/game-mock');

function preprocessGameData(obstacles, speed) {
    if (obstacles.length === 0) {
        return tf.tensor2d([0, 0, 0, speed], [1, 4]);
    }

    const obstacle = obstacles[0];
    return tf.tensor2d([obstacle.xPos, obstacle.width, obstacle.yPos, speed], [1, 4]);
}

// update the target model's weights with the online model's weights
async function updateTargetModel(onlineModel, targetModel) {
    const onlineWeights = onlineModel.getWeights();
    await targetModel.setWeights(onlineWeights);
}

async function trainDinoModel(onlineModel, targetModel, proxy, episodes, memory, batchSize, gamma, epsilonStart, epsilonEnd, epsilonDecay, targetUpdateFrequency) {
    let epsilon = epsilonStart;
    let episodeReward = 0;

    // new session of training
    fs.appendFile('logs.txt', `\n\n------------------------------------------------------------------\n\n`)
        .catch((err) => {
            console.error('Failed to write to logs.txt:', err.message);
        });

    for (let episode = 0; episode < episodes; episode++) {
        await proxy.restart();
        let state = preprocessGameData(await proxy.obstacles(), await proxy.speed());
        let done = false;

        while (!done) {
            const action = await selectAction(onlineModel, state, epsilon);
            performAction(action, proxy);

            await tf.nextFrame();

            const nextState = preprocessGameData(await proxy.obstacles(), await proxy.speed());
            const crashed = await proxy.crashed();
            const reward = computeReward(crashed);
            episodeReward += reward;
            done = crashed;

            memory.add({ state, action, reward, nextState, done }); // Fix the experience structure here
            state = nextState;

            if (memory.buffer.length >= batchSize) {
                const experiences = memory.sample(batchSize);
                const loss = await optimizeModel(onlineModel, targetModel, experiences, gamma, batchSize);
                console.log('Loss:', loss);
            }

            // Decay epsilon (exploration rate)
            epsilon = epsilonEnd + (epsilonStart - epsilonEnd) * Math.exp(-1 * episode / epsilonDecay);
        }

        // Update the target model's weights periodically
        if ((episode + 1) % targetUpdateFrequency === 0) {
            await updateTargetModel(onlineModel, targetModel);
        }

        // log after every episode and set episodeReward to 0
        fs.appendFile('logs.txt', `Episode: ${episode + 1}, Epsilon: ${epsilon}, Total Reward: ${episodeReward}\n`)
            .catch((err) => {
                console.error('Failed to write to logs.txt:', err.message);
            });
        episodeReward = 0;
    }
}

async function optimizeModel(onlineModel, targetModel, experiences, gamma, batchSize) {
    const inputs = [];
    const targets = [];

    for (const { state, action, reward, nextState, done }
        of experiences) {
        const inputTensor = tf.tensor2d([Array.from(state.dataSync())]);
        const onlineModelQValues = onlineModel.predict(inputTensor);

        const nextInputTensor = tf.tensor2d([Array.from(nextState.dataSync())]);
        const onlineModelNextQValues = onlineModel.predict(nextInputTensor);

        const targetInputTensor = tf.tensor2d([Array.from(nextState.dataSync())]);
        const targetModelNextQValues = targetModel.predict(targetInputTensor);

        const reshapedOnlineModelNextQValues = onlineModelNextQValues.reshape([1, -1]);

        // Use tf.tidy() and tf.keep() to manage memory efficiently
        let nextQValue;
        tf.tidy(() => {
            const nextQValueIndex = onlineModelNextQValues.argMax(1);
            const nextQValueTensor = tf.keep(targetModelNextQValues.gather(nextQValueIndex, 1));
            nextQValue = nextQValueTensor.dataSync()[0];
        });

        let targetQValue;
        if (done) {
            targetQValue = reward;
        } else {
            targetQValue = reward + gamma * nextQValue;
        }

        const targetArray = Array.from(onlineModelQValues.dataSync());
        targetArray[action] = targetQValue;
        inputs.push(Array.from(state.dataSync()));
        targets.push(targetArray);

        inputTensor.dispose();
        onlineModelQValues.dispose();
        nextInputTensor.dispose();
        onlineModelNextQValues.dispose();
        targetInputTensor.dispose();
        targetModelNextQValues.dispose();
        reshapedOnlineModelNextQValues.dispose();
    }

    const inputTensor = tf.tensor2d(inputs);
    const targetTensor = tf.tensor2d(targets);

    return onlineModel.fit(inputTensor, targetTensor, { epochs: 1, batchSize, shuffle: true }).then((history) => history.history.loss[0]);
}

async function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [4] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: tf.losses.meanSquaredError
    });

    return model;
}

async function launchBrowser() {
    const browser = await puppeteer.launch({
        headless: false,
        args: ['--mute-audio'] // Add this line to mute audio
    });
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:8080');
    return page;
}

async function setupModelTraining() {
    const page = await launchBrowser();
    await page.waitForFunction('Runner.instance_ !== undefined');
    const gameProxy = await proxy(page);

    // Load the saved model or create a new one if it doesn't exist
    let model;
    try {
        model = await tf.loadLayersModel('file://./dino-chrome-model/main/model.json');
        console.log('Loaded saved model'.green);
        model.compile({ // Compile the loaded model
            optimizer: tf.train.adam(0.001),
            loss: tf.losses.meanSquaredError
        });
    } catch (error) {
        console.log('No saved model found, creating a new one'.yellow);
        model = await createModel();
    }

    // Load the target model from the same file location as the main model
    let targetModel;
    try {
        targetModel = await tf.loadLayersModel('file://./dino-chrome-model/main/model.json');
        console.log('Loaded target model'.green);
        targetModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: tf.losses.meanSquaredError
        });
    } catch (error) {
        console.log('No target model found, cloning the main model'.yellow);
        targetModel = await createModel();
    }

    const episodes = 1000;
    const memory = new Memory(100000);
    const batchSize = 32;
    const gamma = 0.9; // Discount factor
    const epsilonStart = 1.0; // Initial exploration rate
    const epsilonEnd = 0.01; // Final exploration rate
    const epsilonDecay = 200; // Decay rate for exploration
    const targetUpdateFrequency = 50; // How often to update the target model

    const saveAndExit = async() => {
        console.log('Saving models before exit...'.green);
        await model.save('file://./dino-chrome-model/main');
        console.log('Main model saved successfully.'.green);
        await targetModel.save('file://./dino-chrome-model/target');
        console.log('Target model saved successfully.'.green);
        process.exit();
    };

    // Handle SIGINT (Ctrl + C) using readline
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    rl.on('SIGINT', async() => {
        rl.close(); // Close the readline interface
        await saveAndExit(); // Save the model and exit
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', async(error) => {
        console.error(`Uncaught exception: ${error.message.red}`);
        await saveAndExit();
    });

    // Train the model here (additionally or for the first time)
    try {
        await trainDinoModel(model, targetModel, gameProxy, episodes, memory, batchSize, gamma, epsilonStart, epsilonEnd, epsilonDecay, targetUpdateFrequency);
    } catch (error) {
        console.error(`Error during training: ${error}`);
    }

    // Save the models to disk storage in separate folders
    await model.save('file://./dino-chrome-model/main');
    await targetModel.save('file://./dino-chrome-model/target');
}

module.exports = {
    setupModelTraining
};