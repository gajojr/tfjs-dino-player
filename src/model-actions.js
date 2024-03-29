const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');
const readline = require('readline');
const colors = require('colors');
const fs = require('fs').promises;
const Memory = require('../src/Memory');
const { ChromeGameProxy } = require('../src/game-mock');
// const NoisyDense = require('./NoisyDense');
const AdvantageNormalization = require('./AdvantageNormalization');

// Function to compute the Kullback-Leibler Divergence between two probability distributions
function klDivergence(yTrue, yPred) {
    // tf.tidy() is used to free any intermediate tensors created during this function
    return tf.tidy(() => {
        // Clipping yTrue and yPred values to avoid division by zero and log(0) issues
        const yTrueClipped = tf.clipByValue(yTrue, 1e-7, 1);
        const yPredClipped = tf.clipByValue(yPred, 1e-7, 1);

        // Division of yTrueClipped by yPredClipped
        const division = yTrueClipped.div(yPredClipped);

        // Taking the logarithm of the division result
        const logDiv = tf.log(division);

        // Multiplying logDiv with yTrueClipped and summing over the last axis to get the KL Divergence
        const klDiv = logDiv.mul(yTrueClipped).sum(-1);

        // Returning the KL Divergence tensor
        return klDiv;
    });
}

// Function to compute the sum of KL Divergence values over the last axis
function sum_kl(yTrue, yPred) {
    // Calling klDivergence function and then summing over the last axis
    return klDivergence(yTrue, yPred).sum(-1);
}

// Function to clip a value v between a minimum and maximum value
function clip(v, min, max) {
    // Using Math.min and Math.max to clip the value v between min and max
    return Math.min(Math.max(v, min), max);
}

class DinoAgent {
    constructor(proxy) {
        this.proxy = proxy;
        this.memory = new Memory(100000);
        // Batch size.
        this.batchSize = 32;
        // Discount factor (0 < gamma <= 1).
        this.gamma = 0.9;
        // How often to update the target model, must be multiple of batch size.
        this.targetUpdateFrequency = 1280;
        // Exploration <-> exploitation
        this.epsilon = 1;
        this.epsilonDecay = 0.01;
        // Number of steps, used internally to regulate training
        // and target model updates.
        this.steps = 0;
        this.multiSteps = 3;
        // distributional DQN related values
        this.d_atoms = 51; // number of atoms
        this.Vmin = -5; // minimum expected reward
        this.Vmax = +5; // maximum expected reward
        this.delta_z = (this.Vmax - this.Vmin) / (this.d_atoms - 1);
        this.zi = []; // zi array
        for (let i = 0; i < this.d_atoms; i++) {
            this.zi.push(this.Vmin + i * this.delta_z);
        }
    }

    obstacleToVector(obstacle, vector, offset) {
        // Minimum distance (crash): 19
        // Maximum distance (appear): 625
        // Convert distance into int from 0-39.
        const distance = Math.min(39, Math.max(0, ~~((obstacle.xPos - 19) / 16)));
        vector[offset + distance] = 1;
        vector[offset + 40] = obstacle.yPos;
        vector[offset + 41] = obstacle.width;
        vector[offset + 42] = obstacle.size;
    }

    stateToVector(state) {
        // shape of tensor: (106,)
        // [0-39]: distance to obstacle 0, one-hot encoded
        // [40]: y-position of obstacle 0
        // [41]: width of obstacle 0
        // [42]: height of obstacle 0
        // [43-85]: same for obstacle 1
        // [86]: jumping? 0/1
        // [87]: y-position of t-rex
        // [88-105]: current game speed, one-hot encoded
        let vector = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ];

        const obstacles = state.obstacles;
        if (obstacles.length > 0) {
            this.obstacleToVector(obstacles[0], vector, 0);

            if (obstacles.length > 1) {
                this.obstacleToVector(obstacles[1], vector, 43);
            }
        }

        // Jump yes/no:
        if (state.jumping) {
            vector[86] = 1;
        }
        vector[87] = state.ypos;
        let speed = Math.min(17, Math.max(0, Math.round(state.speed - 6)));
        vector[88 + speed] = 1;

        return vector;
    }

    async delay(time) {
        return new Promise((resolve) => setTimeout(resolve, time));
    }

    // update the target model's weights with the online model's weights
    async updateTargetModel() {
        const onlineWeights = this.onlineModel.getWeights();
        await this.targetModel.setWeights(onlineWeights);
        //await this.onlineModel.save('file://./dino-chrome-model/main');
        await this.targetModel.save('file://./dino-chrome-model/target');
    }

    async trainModel(episodes) {
        // new session of training
        fs.appendFile(
            'logs.txt',
            `\n\n------------------------------------------------------------------\n\n`
        ).catch((err) => {
            console.error('Failed to write to logs.txt:', err.message);
        });

        let highScore = 0; // Variable to keep track of the high score across episodes

        for (let episode = 0; episode < episodes; episode++) { // Looping through each episode
            let episodeReward = 0; // Variable to accumulate the total reward for the episode
            let episodeSteps = 0; // Variable to count the steps taken in the episode

            await this.proxy.restart();
            await this.delay(100); // give game time to settle
            await this.proxy.jump(); // to trigger start of game

            let state = await this.proxy.state(); // Getting the initial game state

            let multiStepBuffer = []; // Buffer to store multiple steps for n-step Q-learning

            while (!state.done) { // While the game is not over
                const stateVector = this.stateToVector(state); // Converting state to a vector
                const action = this.selectAction(stateVector); // Selecting an action based on the state

                this.proxy.performAction(action); // Performing the selected action

                if (episodeSteps > 10000) {
                    console.log(this.proxy.str());
                }

                // We need some time to pass, otherwise the next state will
                // be too close to the current state and model won't see the
                // result of it's action.
                // Instead of sleeping, we can do the optimization here. This
                // will take some time.
                if (this.memory.count >= this.batchSize) {
                    await this.optimizeModel();
                }

                // Chrome game seems to use 60 FPS, so time in game passes
                // in steps of about 16-17 ms. We try to use 30 FPS, so
                // we wait here until two time steps (about 30-40 ms) passed.
                let nextState = await this.proxy.state();
                while (!nextState.done && nextState.time - state.time < 30) {
                    await this.delay(5);
                    nextState = await this.proxy.state();
                }

                const timeDelta = nextState.time - state.time; // Calculating the time difference between states

                // Evaluate this step if and only if in time frame.
                if (timeDelta > 40) {
                    // too late, ignore this result because it is
                    // not related to any model action
                    console.warn('Time delta > 40ms: ' + timeDelta);
                } else if (timeDelta < 30) {
                    if (!nextState.done) {
                        // this result should not happen because that means
                        // that delay method failed for an unknown reason
                        console.error('Time delta < 30ms and not done, failed delay?: ' + timeDelta);
                    } else {
                        // terminal state was reached in less than 30 ms.
                        // To not loose this result, we assume that this is
                        // a (late) result of the former action, so last
                        // action is updated. This works only with multi
                        // step buffers.
                        if (multiStepBuffer.length > 0) {
                            // update last entry to be terminal with reward -1
                            const last = multiStepBuffer[multiStepBuffer.length - 1];
                            last.done = true;
                            last.reward = -1;
                            while (multiStepBuffer.length > 0) {
                                this.multiStep(multiStepBuffer);
                            }
                        } else {
                            console.warn("Lost terminal state due to missing multi step buffer.");
                        }
                    }
                } else {
                    const nextStateVector = this.stateToVector(nextState);

                    let reward;
                    if (nextState.done) {
                        reward = -1;
                    } else if (action === 1) {
                        /* experiment: make jumps expensive to avoid random jumps */
                        reward = 0;
                        episodeSteps++;
                    } else {
                        reward = 0.1;
                        episodeSteps++;
                    }
                    episodeReward += reward;

                    multiStepBuffer.push({
                        state: stateVector,
                        action,
                        reward,
                        nextState: nextStateVector,
                        done: nextState.done,
                    });
                    if (nextState.done) {
                        // flush multi step buffer to replay buffer
                        while (multiStepBuffer.length > 0) {
                            this.multiStep(multiStepBuffer);
                        }
                    } else if (multiStepBuffer.length === this.multiSteps) {
                        // add one step from multi step buffer to replay buffer
                        this.multiStep(multiStepBuffer);
                    }
                }

                state = nextState; // Updating the state for the next iteration
            }

            this.epsilon = Math.max(this.epsilon - this.epsilonDecay, 0); // Updating the exploration rate

            highScore = Math.max(highScore, episodeSteps); // Updating the high score if necessary

            // log after every episode and set episodeReward to 0
            console.log("Episode " + episode + " done, score: " + episodeSteps + " (" + highScore + ")");
            fs.appendFile(
                'logs.txt',
                `Episode: ${episode + 1
				}, Total Reward: ${episodeReward}\n`
            ).catch((err) => {
                console.error('Failed to write to logs.txt:', err.message);
            });
        }
    }

    multiStep(multiStepBuffer) {
        // Calculating multi-step reward by summing up the rewards of each step,
        // discounted by gamma raised to the power of the step index.
        let r = 0; // Initialize the cumulative reward to 0
        for (let i = 0; i < multiStepBuffer.length; i++) {
            r += Math.pow(this.gamma, i) * multiStepBuffer[i].reward; // Accumulate the discounted reward
        }

        const first = multiStepBuffer[0]; // Getting the first step from the multi-step buffer
        const last = multiStepBuffer[multiStepBuffer.length - 1]; // Getting the last step from the multi-step buffer

        // Adding a new experience to the memory with the cumulative reward,
        // and the state and action from the first step and the next state and done flag from the last step.
        this.memory.add({
            state: first.state,
            action: first.action,
            reward: r,
            nextState: last.nextState,
            done: last.done,
        });

        // Removing the oldest step from the multi-step buffer as it has been used.
        multiStepBuffer.splice(0, 1);
    }

    async optimizeModel() {
        if (this.steps % this.batchSize !== 0) { // This condition ensures that the optimization is performed only once every batchSize steps for performance reasons.
            // train all batchSize steps only (performance)
            this.steps++;
            return;
        }

        // Sampling a batch of experiences from memory.
        const { samples, sampleIndices } = this.memory.sample(this.batchSize);

        // To speed up, make predictions on full batch:
        const statesVectors = samples.map(s => s.state);
        const nextStatesVectors = samples.map(s => s.nextState);
        const statesTensor = tf.tensor2d(statesVectors);
        const nextStatesTensor = tf.tensor2d(nextStatesVectors);

        // Getting Q-value predictions for current and next states from both the online and target models.
        const onlineModelQValuesTensor = this.onlineModel.apply(statesTensor);
        const onlineModelNextQValuesTensor = this.onlineModel.apply(nextStatesTensor);
        const targetModelNextQValuesTensor = this.targetModel.apply(nextStatesTensor);

        // Converting tensors to JavaScript arrays for further processing.
        const onlineModelQValues = onlineModelQValuesTensor.dataSync();
        const targetModelNextQValues = targetModelNextQValuesTensor.dataSync();

        const noOfActions = targetModelNextQValuesTensor.shape[1];
        const noOfAtomsPerSample = noOfActions * this.d_atoms;

        let tdErrorSum = 0;
        let tdErrorTerminalsSum = 0;
        let terminalsCount = 0;

        // DQN: target model is used to select action and it's prediction
        //      is used as updated value.
        // DDQN: online model is used to select action, but still target 
        //       model's Q value prediction is used as updated value for 
        //       online model.
        // Distributional DQN: predictions are not values but value
        //                     distributions and TD error is set to
        //                     categorical crossentropy loss between
        //                     predicted distribution and updated
        //                     distribution

        // onlineModelNextQValuesTensor has shape (64, 3, 51)
        // multiply by Zi in axis 2, sum up in axis 2 and argmax in axis 1
        const t1 = onlineModelNextQValuesTensor.mul(this.zi);
        const t2 = t1.sum( /* axis */ 2);
        const t3 = tf.argMax(t2, /* axis */ 1)
        const nextActions = t3.dataSync();
        let priorities = [];
        for (let i = 0; i < this.batchSize; i++) {
            const r = samples[i].reward;
            const action = samples[i].action;
            const terminal = samples[i].done;
            const nextAction = nextActions[i];

            const p = targetModelNextQValues.slice(
                i * noOfAtomsPerSample + nextAction * this.d_atoms,
                i * noOfAtomsPerSample + (nextAction + 1) * this.d_atoms
            );
            // Categorical algorithm [https://arxiv.org/pdf/1707.06887.pdf]
            // Initializing an array 'm' to store the updated probability mass function (PMF)
            const m = [];
            for (let n = 0; n < this.d_atoms; n++) {
                m[n] = 0;
            }
            for (let j = 0; j < this.d_atoms; j++) {
                // Compute the projection of Tzj onto the support {zi}
                // Tzj is the clipped, discounted, and shifted next-state value for atom j
                let Tzj;
                if (!terminal) {
                    // If not a terminal state, calculate Tzj using the Bellman equation
                    Tzj = clip(r + this.gamma * this.zi[j], this.Vmin, this.Vmax);
                } else {
                    // If a terminal state, the reward 'r' is the final value
                    Tzj = clip(r, this.Vmin, this.Vmax);
                }
                // bj is the normalized and shifted value of Tzj
                const bj = (Tzj - this.Vmin) / this.delta_z;
                // l and u are the lower and upper bounds of the index in the support
                const l = Math.floor(bj);
                let u = Math.ceil(bj);
                // If l and u are the same (which is a rare case), increment u to ensure they are different
                if (l == u) {
                    u += 1;
                }
                // Distributing the probability mass of atom j between indices l and u in the updated PMF
                // This is done according to the proximity of bj to l and u
                m[l] += p[j] * (u - bj);
                m[u] += p[j] * (bj - l);
            }


            const index = i * noOfAtomsPerSample + action * this.d_atoms;

            const originalDistribution = onlineModelQValues.slice(
                index,
                index + this.d_atoms
            );
            // TD error is klDivergence between target and original
            const tdError = tf.tidy(
                () => klDivergence(tf.tensor(m), tf.tensor(originalDistribution)).dataSync()[0]
            );

            // re-integrate target values into onlineModelQValues
            for (let n = 0; n < this.d_atoms; n++) {
                onlineModelQValues[index + n] = m[n];
            }

            tdErrorSum += tdError;
            if (terminal) {
                tdErrorTerminalsSum += tdError;
                terminalsCount++;
            }

            priorities.push(tdError);
        }

        console.log("Avg. TD error: " + (tdErrorSum / this.batchSize) + ", terminals only: " + (tdErrorTerminalsSum / terminalsCount));

        // Set TD error as new priority for the samples in the memory
        this.memory.updatePriorities(sampleIndices, priorities);

        const targetTensor = tf.tensor3d(onlineModelQValues, onlineModelQValuesTensor.shape);

        await this.onlineModel.fit(statesTensor, targetTensor, { epochs: 1, batchSize: this.batchSize, shuffle: true, verbose: false });

        // Update the target model's weights periodically
        if (this.steps % this.targetUpdateFrequency === 0) {
            console.log("Updating target model.");
            await this.updateTargetModel();
        }

        statesTensor.dispose();
        nextStatesTensor.dispose();
        targetTensor.dispose();
        onlineModelQValuesTensor.dispose();
        onlineModelNextQValuesTensor.dispose();
        targetModelNextQValuesTensor.dispose();

        this.steps++;
    }

    createModel() {
        /* 
         * Model is a distributional dueling DDQN. 
         * It had two branches, one for state value distribution (V) 
         * and one for advance distribution (A).
         * Value distribution has shape (1,D_ATOMS), advance
         * distribution has shape (3,D_ATOMS). Number of atoms
         * usually is 51 ("C51" model).
         * Both branches are recombined into a single distribution
         * (Q) with shape (3,D_ATOMS).
         */
        const i = tf.input({ shape: [106], name: 'state_input' });
        const cd1 = tf.layers.dense({ units: 64, activation: 'relu', name: 'common_dense1' }).apply(i);
        const cd2 = tf.layers.dense({ units: 64, activation: 'relu', name: 'common_dense2' }).apply(cd1);
        // now break up into two branches:
        const x1 = tf.layers.dense({ units: 64, activation: 'relu', name: 'v_dense' }).apply(cd2);
        const x2 = tf.layers.dense({ units: 64, activation: 'relu', name: 'a_dense' }).apply(cd2);
        const v_flat = tf.layers.dense({ units: this.d_atoms, activation: 'linear', name: 'flat_v' }).apply(x1);
        // reshape V to (1, D_ATOMS)
        const v = tf.layers.reshape({ targetShape: [1, this.d_atoms], name: 'reshaped_v' }).apply(v_flat);
        const a_flat = tf.layers.dense({ units: 3 * this.d_atoms, activation: 'linear', name: 'flat_a' }).apply(x2);
        // reshape A to (3, D_ATOMS)
        const a = tf.layers.reshape({ targetShape: [3, this.d_atoms], name: 'reshaped_a' }).apply(a_flat);
        // and re-merge with a lambda layer
        const merged = new AdvantageNormalization().apply([v, a]);
        // shape of merged data: (3, D_ATOMS)
        const output = tf.layers.softmax({ axis: 2, name: 'softmax_distributions' }).apply(merged);

        const model = tf.model({
            inputs: i,
            outputs: output
        });

        model.compile({
            optimizer: tf.train.adam(0.000125),
            loss: sum_kl,
        });

        return model;
    }

    selectAction(state) {
        const r = Math.random();
        if (r < this.epsilon) {
            // choose random action
            return Math.floor(Math.random() * 3);
        }

        return tf.tidy(() => {
            // Choose the best action according to the model.
            const stateTensor = tf.tensor2d(state, [1, 106]);
            const pDist = this.onlineModel.apply(stateTensor);
            // pDist is per-action probability distribution over 
            // possible rewards, shape (null, 3, 51)
            // Q(s,a) value is sum(Zi * p(i)), shape (null, 3,)
            const q = pDist.mul(this.zi).sum( /* axis */ 2);
            // use argmax as usual to determine most valuable action
            const action = tf.argMax(q, /* axis= */ 1);
            return action.dataSync()[0];
        });
    }
}

async function launchBrowser() {
    const browser = await puppeteer.launch({
        headless: false,
        args: ['--mute-audio'], // Add this line to mute audio
    });
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:8080');
    return page;
}

async function createChromeGameProxy() {
    const page = await launchBrowser();
    await page.waitForFunction('Runner.instance_ !== undefined');
    return gameProxy = new ChromeGameProxy(page);
}

function compileModel(model) {
    model.compile({
        optimizer: tf.train.adam(0.000125),
        loss: sum_kl,
    });
}

async function setupModelTraining() {
    const gameProxy = await createChromeGameProxy();

    const dinoAgent = new DinoAgent(gameProxy);

    // Load the saved model or create a new one if it doesn't exist
    try {
        dinoAgent.targetModel = await tf.loadLayersModel(
            'file://./dino-chrome-model/target/model.json'
        );
        compileModel(dinoAgent.targetModel);
        console.log('Loaded saved model'.green);
        // important: to see effect of loaded model,
        // set epsilon to 0 to be greedy, otherwise
        // model will fail fast due to epsilon == 1.
        dinoAgent.epsilon = 0;
    } catch (error) {
        console.log('No saved model found, creating a new one'.yellow);
        dinoAgent.targetModel = dinoAgent.createModel();
    }

    dinoAgent.onlineModel = dinoAgent.createModel();
    dinoAgent.onlineModel.setWeights(dinoAgent.targetModel.getWeights());
    compileModel(dinoAgent.onlineModel);

    const episodes = 100000;

    // Train the model here (additionally or for the first time)
    await dinoAgent.trainModel(episodes);
}

async function setupModelPlaying() {
    const gameProxy = await createChromeGameProxy();

    const dinoAgent = new DinoAgent(gameProxy);

    // Load the saved model
    try {
        dinoAgent.targetModel = await tf.loadLayersModel(
            'file://./dino-chrome-model/target/model.json'
        );
        compileModel(dinoAgent.targetModel);
        console.log('Loaded saved model'.green);
        dinoAgent.epsilon = 0; // setting epsilon to 0 to make the model choose the best action always
    } catch (error) {
        console.error('No saved model found. You need to train the model first.'.red);
        // You might want to exit the process if there's no model to play with
        process.exit(1);
    }

    dinoAgent.onlineModel = dinoAgent.createModel();
    dinoAgent.onlineModel.setWeights(dinoAgent.targetModel.getWeights());
    compileModel(dinoAgent.onlineModel);

    // Now enter the playing loop
    await gameProxy.restart(); // Restart the game to ensure a fresh start
    await dinoAgent.delay(100); // give game time to settle
    await gameProxy.jump(); // to trigger start of game

    let state = await gameProxy.state();
    while (true) { // Infinite loop to keep playing
        const stateVector = dinoAgent.stateToVector(state);
        const action = dinoAgent.selectAction(stateVector);

        gameProxy.performAction(action); // Perform the selected action

        let nextState = await gameProxy.state();
        while (!nextState.done && nextState.time - state.time < 30) {
            await dinoAgent.delay(5);
            nextState = await gameProxy.state();
        }

        state = nextState; // Update the state for the next iteration

        if (state.done) {
            // If the game ended, restart and continue playing
            await gameProxy.restart();
            await dinoAgent.delay(100); // give game time to settle
            await gameProxy.jump(); // to trigger start of game
            state = await gameProxy.state();
        }
    }
}

module.exports = {
    setupModelTraining,
    setupModelPlaying,
};