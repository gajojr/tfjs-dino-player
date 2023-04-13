const tf = require('@tensorflow/tfjs-node');

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

module.exports = Memory;