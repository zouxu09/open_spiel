use rust_open_spiel::*;

pub fn play_tic_tac_toe() {
    let game = Game::new("tic_tac_toe");
    println!("The short name is: {}", game.short_name());
    println!("The long name is: {}", game.long_name());
    println!("Number of players: {}", game.num_players());
    println!("Number of distinct actions: {}", game.num_distinct_actions());
    println!("Max game length: {}", game.max_game_length());

    let state = game.new_initial_state();
    println!("Initial state:\n{}", state.to_string());

    let clone = state.clone();
    println!("Cloned initial state:\n{}", clone.to_string());

    while !state.is_terminal() {
        println!("");
        println!("State:\n{}", state.to_string());
        let legal_actions = state.legal_actions();
        let player = state.current_player();
        println!("Legal actions: ");
        let action = legal_actions[0];
        for a in legal_actions {
            println!("  {}: {}", a, state.action_to_string(player, a));
        }
        println!("Taking action {}: {}", action, state.action_to_string(player, action));
        state.apply_action(action);
    }

    println!("Terminal state reached:\n{}\n", state.to_string());
    let returns = state.returns();
    for i in 0..game.num_players() {
        println!("Utility for player {} is {}", i, returns[i as usize]);
    }
}

pub fn play_tic_tac_toe_with_bots() {
    let game = Game::new("tic_tac_toe");
    let state = game.new_initial_state();
    println!("Initial state:\n{}", state.to_string());

    let mut params = GameParameters::default();
    params.set_int("seed", 42);

    let mut bots = vec![
        create_bot_by_name("uniform_random", &game, 0, &params),
        create_bot_by_name("uniform_random", &game, 1, &params),
    ];

    for _ in 0..2 {
        while !state.is_terminal() {
            let player = state.current_player();
            let action = bots[player as usize].step(&state);
            let enemy = 1 - player;
            bots[enemy as usize].inform_action(&state, player, action);
            state.apply_action(action);
        }
        for bot in bots.iter_mut() {
            bot.restart();
        }
    }

    println!("Terminal state reached:\n{}\n", state.to_string());
}

#[test]
fn tic_tac_toe_test() {
    play_tic_tac_toe();
}

#[test]
fn tic_tac_toe_with_bots_test() {
    play_tic_tac_toe_with_bots();
}

#[test]
fn new_game_with_parameters_test() {
    let mut params = GameParameters::default();
    params.set_str("name", "go");
    params.set_int("board_size", 9);
    params.set_f64("komi", 7.5);
    let game = Game::new_with_parameters(&params);
    assert_eq!(
        params.serialize(),
        "board_size=kInt***9***false|||komi=kDouble***7.5***false|||name=kString***go***false"
    );
    assert_eq!(game.short_name(), "go");
    assert_eq!(game.observation_shape(), vec![4, 9, 9]);
}

fn main() {
    play_tic_tac_toe();
    play_tic_tac_toe_with_bots();
}
