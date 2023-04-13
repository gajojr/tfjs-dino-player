const colors = require('colors');
const { setupModelTraining } = require('./src/model-actions');
const { playTheGame } = require('./src/game-mock');

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