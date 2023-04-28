const tf = require('@tensorflow/tfjs-node');

class AdvantageNormalization extends tf.layers.Layer {
	constructor() {
		super({});
	}

	call(inputs) {
		const adv = inputs[0];
		const val = inputs[1];
		const mean = tf.mean(adv, 1, true);
		const centered = adv.sub(mean);
		return centered.add(val);
	}

	getClassName() {
		return 'AdvantageNormalization';
	}
}

module.exports = AdvantageNormalization;
