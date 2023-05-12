const puppeteer = require('puppeteer');

class ChromeGameProxy {
    constructor(page) {
        this.page = page;
        this.inJump = false;
        this.inDuck = false;
    }

    async jump() {
        if (this.inDuck) {
            await this.page.keyboard.up('ArrowDown');
            this.inDuck = false;
        }
        if (!this.inJump) {
            await this.page.keyboard.down('Space');
            this.inJump = true;
        }
    }

    async duck() {
        if (this.inJump) {
            await this.page.keyboard.up('Space');
            this.inJump = false;
        }
        if (!this.inDuck) {
            await this.page.keyboard.down('ArrowDown');
            this.inDuck = true;
        }
    }

    async stand() {
        if (this.inJump) {
            await this.page.keyboard.up('Space');
            this.inJump = false;
        }
        if (this.inDuck) {
            await this.page.keyboard.up('ArrowDown');
            this.inDuck = false;
        }
    }

    async state() {
        return await this.page.evaluate(() => {
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
    async restart() {
      this.page.evaluate(() => {
        Runner.instance_.tRex.xPos = 50;
        Runner.instance_.restart(); 
      });
    } 

    async performAction(action, proxy) {
        switch (action) {
            case 1:
                await this.jump();
                break;
            case 2:
                await this.duck();
                break;
            case 0:
            default:
                await this.stand();
                break;
        }
    }
}

module.exports = {
    ChromeGameProxy
}