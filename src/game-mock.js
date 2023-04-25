const puppeteer = require('puppeteer');

const proxy = page => {
    const jump = async() => {
        await page.keyboard.down('Space');
        await page.keyboard.up('Space');
    };

    const duck = async() => await page.keyboard.down('ArrowDown');

    const stand = async() => await page.keyboard.up('ArrowDown');

    const state = async() => {
        return await page.evaluate(() => {
            i = Runner.instance_;
            return {
                speed: i.currentSpeed,
                jumping: i.tRex.jumping,
                ypos: i.tRex.yPos,
                done: i.crashed,
                obstacles: i.horizon.obstacles,
                time: i.time
            }
        });
    }

    // due to a bug in the game, xPos will increase over time,
    // so reset to 50 here
    const restart = async() => await page.evaluate(() => {
        Runner.instance_.tRex.xPos = 50;
        Runner.instance_.restart(); 
    });

    return { jump, duck, stand, restart, state };
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

module.exports = {
    performAction,
    proxy
}