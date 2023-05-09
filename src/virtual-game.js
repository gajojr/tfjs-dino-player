class Obstacle {
    /*
       An obstacle is a rectangular box with width and height
       and it has a y position (0 for cactus, > 0 for bird).
       Obstacles move by decreasing their distance by 1. Birds
       do not additionally move in x direction, they behave
       like a flying cactus.
    */
    constructor(distance, y, w, h) {
        this.distance = distance
        this.w = w
        this.h = h
        this.y = y
    }
}

function randint(a, b) {
    return Math.floor(Math.random()*(b-a)) + a;
}

const NUMBER_OF_PLAYER_ACTIONS = 3;
const NUMBER_OF_ENV_ACTIONS = 9; // 8 obstacles + pass

const STAND = 0;
const JUMP = 1;
const CROUCH = 2;

const ENV_PASS_ACTION = 3

const JUMP_PHASES = [ 0, 1, 2, 3, 2, 1 ];

const OBSTACLES = [
    [0, 1,2], // big cactus
    [0, 1,1], // small cactus
    [0, 2,2], // two big cactus 
    [0, 2,1], // two small cactus
    [0, 3,2], // three big cactus
    [0, 3,1], // three small cactus
    [1, 1,2], // low bird
    [2, 1,2], // high bird
];

class State {
    /*
      Legal actions for player:
      0: stand
      1: crouch
      2: jump

      Crouching starts immediately and lasts
      as long as crouch action is chosen. Jump
      starts immediately but lasts for some
      frames and during an active jump, the
      only possible action is jump. All other
      actions are considered legal but do not
      affect state.

      To be a MDP, acting player alters between
      player and environment. Environment
      actions are:
      3: nothing happens
      >= 4: obstacle appears (see OBSTACLES)
      Obstacles appear 40 time steps ahead.  
    */
    constructor() {
        this.restart();
    }
    
    restart() {
        this.y = 0; // dino on ground
        this.h = 2; // not crouching
        this.jump_phase = -1; // not jumping
        this.obstacles = [this._create_obstacle(1)];
        this.speed = 100; // 100 ms per step
        this.at_move = 1; // player acts first
        this.score = 0;
        this.terminal = false;
        this.time = 0;
    }

    jump() {
        /* dummy method, used by model to start game */
    }

    clone() {
        const s = new State();
        s.y = this.y;
        s.h = this.h;
        s.jump_phase = this.jump_phase;
        s.speed = this.speed;
        s.at_move = this.at_move;
        s.score = this.score;
        s.terminal = this.terminal;
        s.time = this.time;
        s.obstacles = [];
        this.obstacles.map(o => new Obstacle(o.distance, o.y, o.w, o.h));
        return s;
    }

    _player_action(action) {
        if (this.at_move === 0) {
            throw "player action but environment is at move";
        }
        if (this.jump_phase === -1) { // else: ignore action
            this.h = 2;
            if (action == CROUCH) {
                this.h = 1;
            }
            else if (action == JUMP) {
                this.jump_phase = 0; // # start to jump
                this.y = JUMP_PHASES[this.jump_phase];
            }
        }
        this.at_move = 0; // env next
    }

    _env_action(action) {
        if (this.at_move === 1) {
            throw "environment action but player is at move";
        }
        if (action !== ENV_PASS_ACTION) {
            const od = OBSTACLES[action - 4];
            this.obstacles.push(new Obstacle(40, od[0], od[1], od[2]));
        }
        // generic stuff: jump progress, time progress
        if (this.jump_phase !== -1) {
            this.jump_progress();
        }
        this.time_progress();
        this.at_move = 1; // player next
    }

    jump_progress() {
        this.jump_phase += 1
        if (this.jump_phase === JUMP_PHASES.length) {
            this.y = 0;
            this.jump_phase = -1;
        }
        else {
            this.y = JUMP_PHASES[this.jump_phase];
        }
    }

    time_progress() {
        const obstacles = this.obstacles;
        this.obstacles = [];
        for (const obstacle of obstacles) {
            obstacle.distance -= 1;
            // detect collision
            if (obstacle.distance <= 0 && obstacle.distance + obstacle.w - 1 >= 0) {
                const y1 = obstacle.y;
                const y2 = obstacle.y + obstacle.h - 1;
                const dino_y1 = this.y;
                const dino_y2 = this.y + this.h - 1;
                if ((y1 >= dino_y1 && y1 <= dino_y2) || (y2 >= dino_y1 && y2 <= dino_y2)) {
                    // crash
                    this.terminal = true;
                }
            }
            if (obstacle.distance + obstacle.w + 1 > 0) { 
                this.obstacles.push(obstacle);
            }
        }
        if (!this.terminal) {
            this.score += 1;
        }
    }

    performAction(action) {
        this.apply_action(action);
        // performAction function automatically takes care of
        // environment action
        if (!this.terminal) {
            this.apply_action(this.choose_random_env_action());
        }
        this.time = this.time + 50; // simulate 50 ms per step
    }

    apply_action(action) {
        if (this.terminal) {
          throw "apply_action on terminal state not possible";
        }
        if (action < 0) {
            throw "action out of bounds";
        }
        else if (action < NUMBER_OF_PLAYER_ACTIONS) {
           this._player_action(action);
        }
        else if (action < NUMBER_OF_PLAYER_ACTIONS + NUMBER_OF_ENV_ACTIONS) {
            this._env_action(action);
        }
        else {
            throw "action out of bounds";
        }
    }

    _create_obstacle(max_offset) {
        const idx = randint(0, max_offset);
        const obstacle_data = OBSTACLES[idx];
        return new Obstacle(40, obstacle_data[0], obstacle_data[1], obstacle_data[2])
    }

    choose_random_env_action() {
        if (this.at_move !== 0) {
            throw "env action but env is not at move";
        }
        // random choice strategy:
        // up to 200 points: 0,1, distance 15-25
        // up to 300 points: 0,1,2,3, distance 10-15
        // up to 400 points: 0-6, distance 5-15
        // more than 500 points: all actions, distance 10-20
        let max, min_dist, max_dist;
        if (this.score < 200) {
            max = 1;
            min_dist = 15;
            max_dist = 25;
        }
        else if (this.score < 300) {
            max = 3;
            min_dist = 10;
            max_dist = 15;
        }
        else if (this.score < 400) {
            max = 5;
            min_dist = 5;
            max_dist = 15;
        }
        else {
            max = 7;
            min_dist = 10;
            max_dist = 20;
        }
        max = 7;

        // New obstacles appear in distance 40 from dino.
        // They appear if and only if distance to last
        // obstacle is within min_dist and max_dist.
        const idx = this.obstacles.length - 1;
        const current_dist = 40 - (this.obstacles[idx].distance + this.obstacles[idx].w - 1);
        if (current_dist >= min_dist && current_dist < max_dist) {
            // It may or may not appear.
            // Probability is 1/(max_dist - current_dist), so for
            // example 1/10 if there are 10 distance points left
            // for obstacle to appear.
            const dist_rand = randint(0, max_dist - current_dist);
            if (dist_rand == 0) {
                return randint(0, max) + ENV_PASS_ACTION + 1;
            }
            else {
                return ENV_PASS_ACTION;
            }
        }
        else if (current_dist >= max_dist) {
            // it will appear
            return randint(0, max) + ENV_PASS_ACTION + 1;
        }
        else {
          return ENV_PASS_ACTION; // no new obstacle
        }
    }

    str() {
        const s = [...
        "                                             #\n" +
        "                                             #\n" +
        "                                             #\n" +
        "                                             #\n" +
        ".............................................#\n" ];
        // dino at x=3
        for (let i=0; i<this.h; i++) {
            s[(4-(this.y+i))*47 + 3] = 'D';
        }
        // obstacles
        for (const obstacle of this.obstacles) {
            for (let i=0; i<obstacle.h; i++) {
                for (let j=0; j<obstacle.w; j++) {
                    if (obstacle.y === 0) { // cactus
                        s[(4-(obstacle.y+i))*47 + 3 + obstacle.distance + j] = '|';
                    }
                    else { // bird
                        s[(4-(obstacle.y+i))*47 + 3 + obstacle.distance + j] = '<';
                    }
                }
            }
        }
        return s.join("");
    }

    state() {
        const ypos = this.jump_phase > -1 ? JUMP_PHASES[this.jump_phase] : 0;
        const obstacles = this.obstacles.map(o => { return { xPos: o.distance*16 + 19, yPos: o.y, width: o.w, size: o.h }; });
        const jumping = this.jump_phase > -1;
        const speed = 6;
        const time = this.time;
        const done = this.terminal;
        return {
            speed,
            jumping,
            ypos,
            done,
            obstacles,
            time
        };
    }
}

module.exports = {
    State
}