const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');
const readline = require('readline');
const colors = require('colors');
const fs = require('fs').promises;

class Memory {
    constructor(bufferSize) {
        this.bufferSize = bufferSize;
        this.buffer = [];
        this.pointer = 0;
    }

    add(experience) {
        if (this.buffer.length < this.bufferSize) {
            this.buffer.push(experience);
        } else {
            this.buffer[this.pointer] = experience;
            this.pointer = (this.pointer + 1) % this.bufferSize;
        }
    }

    sample(batchSize) {
        const sampleIndices = tf.util.createShuffledIndices(this.buffer.length);
        const samples = [];
        for (let i = 0; i < batchSize; i++) {
            samples.push(this.buffer[sampleIndices[i]]);
        }
        return samples;
    }
}

function preprocessGameData(obstacles, speed) {
    if (obstacles.length === 0) {
        return tf.tensor2d([0, 0, 0, speed], [1, 4]);
    }

    const obstacle = obstacles[0];
    return tf.tensor2d([obstacle.xPos, obstacle.width, obstacle.yPos, speed], [1, 4]);
}

function computeReward(crashed) {
    return crashed ? -1 : 0.1;
}

async function selectAction(model, state, epsilon) {
    if (Math.random() < epsilon) {
        // Choose a random action with probability epsilon
        return Math.floor(Math.random() * 3);
    } else {
        // Choose the best action according to the model
        const inputTensor = tf.tensor2d([Array.from(state.dataSync())]);
        const qValues = model.predict(inputTensor);
        const action = (await qValues.argMax(-1).data())[0];
        return action;
    }
}

async function trainDinoModel(model, proxy, episodes, memory, batchSize, gamma, epsilonStart, epsilonEnd, epsilonDecay) {
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
            const action = await selectAction(model, state, epsilon);
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
                const loss = await optimizeModel(model, experiences, gamma);
                console.log('Loss:', loss);
            }

            if (done) {
                fs.appendFile('logs.txt', `Episode: ${episode + 1}, Epsilon: ${epsilon}, Total Reward: ${episodeReward}\n`)
                    .catch((err) => {
                        console.error('Failed to write to logs.txt:', err.message);
                    });
                episodeReward = 0;
                // console.log(`Episode: ${episode + 1}, Epsilon: ${epsilon}`);
            }
            // Decay epsilon (exploration rate)
            epsilon = epsilonEnd + (epsilonStart - epsilonEnd) * Math.exp(-1 * episode / epsilonDecay);
        }
    }
}

async function performAction(action, proxy) {
    switch (action) {
        case 1:
            await proxy.jump();
            break;
        case 2:
            await proxy.duck();
            break;
        case 0:
        default:
            await proxy.stand();
            break;
    }
}

function optimizeModel(model, experiences, gamma) {
    const inputs = [];
    const targets = [];

    for (const { state, action, reward, nextState, done }
        of experiences) {
        const inputTensor = tf.tensor2d([Array.from(state.dataSync())]);
        const target = model.predict(inputTensor);
        const nextInputTensor = tf.tensor2d([Array.from(nextState.dataSync())]);
        const nextQValues = model.predict(nextInputTensor);

        let targetQValue;
        if (done) {
            targetQValue = reward;
        } else {
            targetQValue = reward + gamma * nextQValues.max().dataSync()[0];
        }

        const targetArray = Array.from(target.dataSync());
        targetArray[action] = targetQValue;
        inputs.push(state.dataSync());
        targets.push(targetArray);
    }

    const inputTensor = tf.tensor2d(inputs);
    const targetTensor = tf.tensor2d(targets);

    return model.fit(inputTensor, targetTensor, { epochs: 1 }).then((history) => history.history.loss[0]);
}

async function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [4] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
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

const proxy = async(page) => {
    const jump = async() => {
        await page.keyboard.down('Space');
        await page.keyboard.up('Space');
    };

    const duck = async() => await page.keyboard.down('ArrowDown');

    const stand = async() => await page.keyboard.up('ArrowDown');

    const speed = async() => {
        const currentSpeed = await page.evaluate(() => Runner.instance_.currentSpeed);
        return currentSpeed;
    };

    const obstacles = async() => {
        const obstacleData = await page.evaluate(() => Runner.instance_.horizon.obstacles);
        return obstacleData;
    };

    const restart = async() => await page.evaluate(() => Runner.instance_.restart());

    const crashed = async() => {
        const isCrashed = await page.evaluate(() => Runner.instance_.crashed);
        return isCrashed;
    };

    return { jump, duck, stand, speed, obstacles, restart, crashed };
};

// game loop for playing
async function gameLoopPlay(proxy, model) {
    await proxy.restart();
    await proxy.jump(); // jump to start the game

    let state = preprocessGameData(await proxy.obstacles(), await proxy.speed());

    while (true) {
        const inputTensor = tf.tensor2d([Array.from(state.dataSync())]);
        const action = (await model.predict(inputTensor).argMax(-1).data())[0];

        await performAction(action, proxy);

        await tf.nextFrame();

        state = preprocessGameData(await proxy.obstacles(), await proxy.speed());

        if (await proxy.crashed()) {
            await proxy.restart();
            state = preprocessGameData(await proxy.obstacles(), await proxy.speed());
        }
    }
}

async function playTheGame() {
    const page = await launchBrowser();
    await page.waitForFunction('Runner.instance_ !== undefined');
    const gameProxy = await proxy(page);

    // Load the saved model
    let model;
    try {
        model = await tf.loadLayersModel('file://./dino-chrome-model/model.json');
        console.log('Model loaded'.green);
        model.compile({ // Compile the loaded model
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
        });
    } catch (error) {
        console.log(error);
        console.log('No model found, please train the model first'.red);
        process.exit();
    }

    gameLoopPlay(gameProxy, model);
}

async function setupModelTraining() {
    const page = await launchBrowser();
    await page.waitForFunction('Runner.instance_ !== undefined');
    const gameProxy = await proxy(page);

    // Load the saved model or create a new one if it doesn't exist
    let model;
    try {
        model = await tf.loadLayersModel('file://./dino-chrome-model/model.json');
        console.log('Loaded saved model'.green);
        model.compile({ // Compile the loaded model
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
        });
    } catch (error) {
        console.log('No saved model found, creating a new one'.yellow);
        model = await createModel();
    }

    const episodes = 1000;
    const memory = new Memory(100000);
    const batchSize = 32;
    const gamma = 0.99; // Discount factor
    const epsilonStart = 1.0; // Initial exploration rate
    const epsilonEnd = 0.01; // Final exploration rate
    const epsilonDecay = 200; // Decay rate for exploration

    const saveAndExit = async() => {
        console.log('Saving model before exit...'.green);
        await model.save('file://./dino-chrome-model');
        console.log('Model saved successfully.'.green);
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
        await trainDinoModel(model, gameProxy, episodes, memory, batchSize, gamma, epsilonStart, epsilonEnd, epsilonDecay);
    } catch (error) {
        console.error(`Error during training: ${error}`);
    }

    // Save the model to disk storage in the same folder
    await model.save('file://./dino-chrome-model');
}

// pass arg --train or --play
(async() => {
    const args = process.argv.slice(2);

    if (args.includes('--train')) {
        await setupModelTraining();
    } else if (args.includes('--play')) {
        await playTheGame();
    } else {
        console.log(`Please provide an argument: ${'--train'.yellow} or ${'--play'.yellow}`);
        process.exit();
    }
})();