const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');

function preprocessGameData(obstacles, speed) {
    if (obstacles.length === 0) {
        return tf.tensor2d([0, 0, 0, speed], [1, 4]);
    }

    const obstacle = obstacles[0];
    return tf.tensor2d([obstacle.xPos, obstacle.width, obstacle.yPos, speed], [1, 4]);
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

function computeReward(crashed) {
    return crashed ? -1 : 0.1;
}

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

async function launchBrowser() {
    const browser = await puppeteer.launch({
        headless: false,
        args: ['--mute-audio'] // Add this line to mute audio
    });
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:8080');
    return page;
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
            optimizer: tf.train.adam(0.001),
            loss: tf.losses.meanSquaredError,
            metrics: ['accuracy']
        });
    } catch (error) {
        console.log(error);
        console.log('No model found, please train the model first'.red);
        process.exit();
    }

    gameLoopPlay(gameProxy, model);
}

module.exports = {
    playTheGame,
    computeReward,
    selectAction,
    performAction,
    proxy
}