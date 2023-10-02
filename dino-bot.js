const colors = require('colors');
const { setupModelTraining, setupModelPlaying } = require('./src/model-actions');

// pass arg --train or --play
(async() => {
    const args = process.argv.slice(2);

    if (args.includes('--train')) {
        await setupModelTraining();
    } else if (args.includes('--play')) {
        await setupModelPlaying();
    } else {
        console.log(`Please provide an argument: ${'--train'.yellow} or ${'--play'.yellow}`);
        process.exit();
    }
})();