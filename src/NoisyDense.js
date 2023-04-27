const tf = require('@tensorflow/tfjs-node');

class NoisyDense extends tf.layers.Layer {
    constructor(units, name, sigma = 0.5, activation = null, useFactorised = true, useBias = true, kernelRegularizer = null, biasRegularizer = null, activityRegularizer = null, kernelConstraint = null, biasConstraint = null) {
        super({ name });
        this.units = units;
        this.sigma = sigma;
        this.useFactorised = useFactorised;
        this.activation = activation;
        this.useBias = useBias;
        this.kernelRegularizer = kernelRegularizer;
        this.biasRegularizer = biasRegularizer;
        this.activityRegularizer = activityRegularizer;
        this.kernelConstraint = kernelConstraint;
        this.biasConstraint = biasConstraint;
    }

    build(inputShape) {
        this.kernel = this.addWeight('kernel', [inputShape[1], this.units], 'float32', tf.initializers.glorotUniform(), this.kernelRegularizer, true, this.kernelConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.units], 'float32', tf.initializers.zeros(), this.biasRegularizer, true, this.biasConstraint);
        }

        this.kernelStddev = this.addWeight('kernelStddev', [inputShape[1], this.units], 'float32', tf.initializers.constant({ value: this.sigma / Math.sqrt(inputShape[1]) }), null, false);
        if (this.useBias) {
            this.biasStddev = this.addWeight('biasStddev', [this.units], 'float32', tf.initializers.constant({ value: this.sigma }), null, false);
        }

        this.inputSpec = [{ ndim: 2 }];
        this.built = true;
    }

    call(inputs) {
        const activation = this.activation;
        const kernel = this.kernel;
        const bias = this.bias;
        const kernelStddev = this.kernelStddev;
        const biasStddev = this.biasStddev;

        const mean = tf.matMul(inputs, kernel);
        const noiseShape = [inputs.shape[0], this.units];
        const stddev = this.useFactorised ? tf.outer(tf.ones(noiseShape), kernelStddev) : kernelStddev;
        const noise = tf.randomNormal(noiseShape, 0, 1).mul(stddev);
        const noisyWeights = kernel.add(noise);

        const output = tf.matMul(inputs, noisyWeights);

        if (this.useBias) {
            const biasNoiseShape = [inputs.shape[0], this.units];
            const biasStddevs = this.useFactorised ? tf.outer(tf.ones(biasNoiseShape), biasStddev) : biasStddev;
            const biasNoise = tf.randomNormal(biasNoiseShape, 0, 1).mul(biasStddevs);
            const noisyBias = bias.add(biasNoise);
            output.add(noisyBias);
        }

        if (activation !== null) {
            output = activation(output);
        }

        return output;
    }
}

module.exports = NoisyDense;