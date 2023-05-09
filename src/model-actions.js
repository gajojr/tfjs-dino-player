const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');
const readline = require('readline');
const colors = require('colors');
const fs = require('fs').promises;
const Memory = require('../src/Memory');
const { ChromeGameProxy } = require('../src/game-mock');
const { State } = require('../src/virtual-game');
const NoisyDense = require('./NoisyDense');
const AdvantageNormalization = require('./AdvantageNormalization');

// Kullback-Leibler divergence between two probability 
// distributions of a discrete random variable.
function klDivergence(yTrue, yPred) {
	return tf.tidy(() => {
		const yTrueClipped = tf.clipByValue(yTrue, 1e-7, 1);
		const yPredClipped = tf.clipByValue(yPred, 1e-7, 1);
		const division = yTrueClipped.div(yPredClipped);
		const logDiv = tf.log(division);
		const klDiv = logDiv.mul(yTrueClipped).sum(-1);
		return klDiv;
	});
}

function sum_kl(yTrue, yPred) {
  return klDivergence(yTrue, yPred).sum(-1);
}

function clip(v, min, max) {
	return Math.min(Math.max(v, min));
}

class DinoAgent {
	constructor(proxy) {
		this.proxy = proxy;
		this.memory = new Memory(100000);
		// Batch size.
		this.batchSize = 64;
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
		for (let i=0; i<this.d_atoms; i++) {
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
		// shape of tensor: (88,)
		// [0-39]: distance to obstacle 0, one-hot encoded
		// [40]: y-position of obstacle 0
		// [41]: width of obstacle 0
		// [42]: height of obstacle 0
		// [43-85]: same for obstacle 1
		// [86]: jumping? 0/1
		// [87]: y-position of t-rex
		let vector = [
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
		//console.log("State: "+vector);

		return vector;
	}

	async delay(time) {
		return new Promise((resolve) => setTimeout(resolve, time));
	}

	// update the target model's weights with the online model's weights
	async updateTargetModel() {
		const onlineWeights = this.onlineModel.getWeights();
		await this.targetModel.setWeights(onlineWeights);
		await this.onlineModel.save('file://./dino-chrome-model/main');
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

		let highScore = 0;

		for (let episode = 0; episode < episodes; episode++) {
			let episodeReward = 0;
			let episodeSteps = 0;

			await this.proxy.restart();
			await this.delay(100); // give game time to settle
			await this.proxy.jump(); // to trigger start of game
			
			let state = await this.proxy.state();

			let multiStepBuffer = [];

			while (!state.done) {
				const stateVector = this.stateToVector(state);
				const action = this.selectAction(stateVector);
				
				this.proxy.performAction(action);

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

				// target is to spend 40 ms between two game states
				let nextState = await this.proxy.state();
				while (!nextState.done && nextState.time - state.time < 40) {
					await this.delay(5);
					nextState = await this.proxy.state();
				}

				const timeDelta = nextState.time - state.time;

				// Evaluate this step if and only if in time frame.
				if (timeDelta > 60) {
					// too late, ignore this result because it is
					// not related to any model action
					console.warn('Time delta > 60ms: ' + timeDelta);
				}
				else if (timeDelta < 40) {
					if (!nextState.done) {
						// this result should not happen because that means
						// that delay method failed for an unknown reason
						console.error('Time delta < 40ms and not done, failed delay?: ' + timeDelta);
					}
					else {
						// terminal state was reached in less than 40 ms.
						// To not loose this result, we assume that this is
						// a (late) result of the former action, so last
						// action is updated. This works only with multi
						// step buffers.
						if (multiStepBuffer.length > 0) {
							// update last entry to be terminal with reward -1
							const last = multiStepBuffer[multiStepBuffer.length-1];
							last.done = true;
							last.reward = -1;
							while (multiStepBuffer.length > 0) {
								this.multiStep(multiStepBuffer);
							}
						}
						else {
							console.warn("Lost terminal state due to missing multi step buffer.");
						}
					}
				} 
				else {
					const nextStateVector = this.stateToVector(nextState);

					let reward;
					if (nextState.done) {
						reward = -1;
					} else if (action === 1) {
						/* experiment: make jumps expensive to avoid random jumps */
						reward = 0;
						episodeSteps ++;
					} else {
						reward = 0.1;
						episodeSteps ++;
					}
					episodeReward += reward;

					multiStepBuffer.push(
						{
							state: stateVector,
							action,
							reward,
							nextState: nextStateVector,
							done: nextState.done,
						}
					);
					if (nextState.done) {
					  // flush multi step buffer to replay buffer
						while (multiStepBuffer.length > 0) {
							this.multiStep(multiStepBuffer);
						}
					}
					else if (multiStepBuffer.length === this.multiSteps) {
					  // add one step from multi step buffer to replay buffer
						this.multiStep(multiStepBuffer);
					}		
				}

				state = nextState;
			}

			this.epsilon = Math.max(this.epsilon - this.epsilonDecay, 0);

			highScore = Math.max(highScore, episodeSteps);

			// log after every episode and set episodeReward to 0
			console.log("Episode " + episode + " done, score: " + episodeSteps+" ("+highScore+")");
			fs.appendFile(
				'logs.txt',
				`Episode: ${
					episode + 1
				}, Total Reward: ${episodeReward}\n`
			).catch((err) => {
				console.error('Failed to write to logs.txt:', err.message);
			});
		}
	}

	multiStep(multiStepBuffer) {
    // calculate multi step reward
    let r = 0;
    for (let i=0; i<multiStepBuffer.length; i++) {
      r += Math.pow(this.gamma, i) * multiStepBuffer[i].reward;
		}
    const first = multiStepBuffer[0];
    const last = multiStepBuffer[multiStepBuffer.length-1];
		this.memory.add(
			{
				state: first.state,
				action: first.action,
				reward: r,
				nextState: last.nextState,
				done: last.done,
			}
		);
    // remove oldest state
    multiStepBuffer.splice(0, 1);
	}

	async optimizeModel() {
		if (this.steps % this.batchSize !== 0) {
			// train all batchSize steps only (performance)
			this.steps++;
			return;
		}

		const { samples, sampleIndices } = this.memory.sample(this.batchSize);

    // To speed up, make predictions on full batch:
    const statesVectors = samples.map(s => s.state);
    const nextStatesVectors = samples.map(s => s.nextState);
    const statesTensor = tf.tensor2d(statesVectors);
    const nextStatesTensor = tf.tensor2d(nextStatesVectors);

    const onlineModelQValuesTensor = this.onlineModel.apply(statesTensor);
    const onlineModelNextQValuesTensor = this.onlineModel.apply(nextStatesTensor);
    const targetModelNextQValuesTensor = this.targetModel.apply(nextStatesTensor);

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
    const t2 = t1.sum(/* axis */ 2);
    const t3 = tf.argMax(t2, /* axis */ 1)
    const nextActions = t3.dataSync();
		let priorities = [];
    for (let i=0; i<this.batchSize; i++) {
      const r = samples[i].reward;
      const action = samples[i].action;
      const terminal = samples[i].done;
      const nextAction = nextActions[i];

      const p = targetModelNextQValues.slice(
				i * noOfAtomsPerSample + nextAction * this.d_atoms, 
				i * noOfAtomsPerSample + (nextAction + 1) * this.d_atoms
			);
      // Categorical algorithm [https://arxiv.org/pdf/1707.06887.pdf]
      const m = [];
			for (let n=0; n<this.d_atoms; n++) {
				m[n] = 0;
			}
      for (let j=0; j<this.d_atoms; j++) {
        // Compute the projection of Tzj onto the support {zi}
				let Tzj;
        if (!terminal) {
          Tzj = clip(r + this.gamma * this.zi[j], this.Vmin, this.Vmax);
				}
				else {
          Tzj = clip(r, this.Vmin, this.Vmax);
				}
        const bj = (Tzj - this.Vmin) / this.delta_z;
        const l = Math.floor(bj);
        let u = Math.ceil(bj);
        if (l == u) {
          u += 1;
				}
        m[l] += p[j]*(u-bj);
        m[u] += p[j]*(bj-l);
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
			for (let n=0; n<this.d_atoms; n++) {
				onlineModelQValues[index + n] = m[n];
			}

      tdErrorSum += tdError;
      if (terminal) {
        tdErrorTerminalsSum += tdError;
        terminalsCount ++;
			}

			priorities.push(tdError);
		}

		console.log("Avg. TD error: "+(tdErrorSum / this.batchSize)+", terminals only: "+(tdErrorTerminalsSum / terminalsCount));

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

		this.steps ++;
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
		const i = tf.input({ shape: [88], name: 'state_input' });
		const cd1 = tf.layers.dense({ units: 64, activation: 'relu', name: 'common_dense1' }).apply(i);
		const cd2 = tf.layers.dense({ units: 64, activation: 'relu', name: 'common_dense2' }).apply(cd1);
    // now break up into two branches:
		const x1 = tf.layers.dense({ units: 64, activation: 'relu', name: 'v_dense' }).apply(cd2);
		const x2 = tf.layers.dense({ units: 64, activation: 'relu', name: 'a_dense' }).apply(cd2);
    const v_flat = tf.layers.dense({ units: this.d_atoms, activation: 'linear', name: 'flat_v' }).apply(x1);
    // reshape V to (1, D_ATOMS)
    const v = tf.layers.reshape({ targetShape: [ 1, this.d_atoms ], name: 'reshaped_v' }).apply(v_flat);
    const a_flat = tf.layers.dense({ units: 3*this.d_atoms, activation: 'linear', name: 'flat_a' }).apply(x2);
    // reshape A to (3, D_ATOMS)
    const a = tf.layers.reshape({ targetShape: [ 3, this.d_atoms ], name: 'reshaped_a' }).apply(a_flat);
    // and re-merge with a lambda layer
    const merged = new AdvantageNormalization().apply([v, a]);
    // shape of merged data: (3, D_ATOMS)
    const output = tf.layers.softmax({ axis: 2, name: 'softmax_distributions' }).apply(merged);

		const model = tf.model({
			inputs: i,
			outputs: output
		});

		model.compile({
			optimizer: tf.train.adam(0.001),
			loss: sum_kl,
		});
	
		return model;
	}

	selectAction(state) {
		const r = Math.random();
		if (r < this.epsilon) {
			// choose random action
			return Math.floor(Math.random()*3);
		}

		return tf.tidy(() => {
			// Choose the best action according to the model.
			const stateTensor = tf.tensor2d(state, [1, 88]);
			const pDist = this.onlineModel.apply(stateTensor);
			// pDist is per-action probability distribution over 
			// possible rewards, shape (null, 3, 51)
			// Q(s,a) value is sum(Zi * p(i)), shape (null, 3,)
			const q = pDist.mul(this.zi).sum(/* axis */ 2);
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

function createVirtualGame() {
	return new State();
}

async function setupModelTraining() {
	//const gameProxy = createVirtualGame();
	const gameProxy = await createChromeGameProxy();

	const dinoAgent = new DinoAgent(gameProxy);

	// Load the saved model or create a new one if it doesn't exist
	// try {
	// 	dinoAgent.onlineModel = await tf.loadLayersModel(
	// 		'file://./dino-chrome-model/main/model.json'
	// 	);
	// 	console.log('Loaded saved model'.green);
	// } catch (error) {
	// 	console.log('No saved model found, creating a new one'.yellow);
	dinoAgent.onlineModel = dinoAgent.createModel();
	// }

	// Load the target model from the same file location as the main model
	// try {
	// 	dinoAgent.targetModel = await tf.loadLayersModel(
	// 		'file://./dino-chrome-model/main/model.json'
	// 	);
	// 	console.log('Loaded target model'.green);
	// } catch (error) {
	// 	console.log('No target model found, cloning the main model'.yellow);
 	dinoAgent.targetModel = dinoAgent.createModel();
	// }
	dinoAgent.targetModel.setWeights(dinoAgent.onlineModel.getWeights());

	const episodes = 100000;

	const saveAndExit = async () => {
		console.log('Saving models before exit...'.green);
		await dinoAgent.onlineModel.save('file://./dino-chrome-model/main');
		console.log('Main model saved successfully.'.green);
		await dinoAgent.targetModel.save('file://./dino-chrome-model/target');
		console.log('Target model saved successfully.'.green);
		process.exit();
	};

	// Handle SIGINT (Ctrl + C) using readline
	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

	rl.on('SIGINT', async () => {
		rl.close(); // Close the readline interface
		await saveAndExit(); // Save the model and exit
	});

	// Handle uncaught exceptions
	// process.on('uncaughtException', async (error) => {
	// 	console.error(`Uncaught exception: ${error.message.red}`);
	// 	//await saveAndExit();
	// });

	// Train the model here (additionally or for the first time)
	// try {
		await dinoAgent.trainModel(episodes);
	// } catch (error) {
	// 	console.error(`Error during training: ${error}`);
	// }

	// Save the models to disk storage in separate folders
	await dinoAgent.onlineModel.save('file://./dino-chrome-model/main');
	await dinoAgent.targetModel.save('file://./dino-chrome-model/target');
}

module.exports = {
	setupModelTraining,
};
