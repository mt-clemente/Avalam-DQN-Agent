use pyo3::prelude::*;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn get_outcomes(states: Vec<[[i32; 9]; 9]>) -> Vec<[[i32; 9]; 9]> {

    let mut res = Vec::with_capacity(264);

    
    for st in states {
        res.append(&mut resulting_states(st));
    };
    

    res
}

fn resulting_states(state: [[i32; 9]; 9]) -> Vec<[[i32; 9]; 9]>{


    let mut res = Vec::with_capacity(264);

    let actions = get_actions(&state);

    for a in actions {
        res.push(play_action(state.clone(),a));
    }

    res
    
}

fn get_actions(state: &[[i32; 9]; 9]) -> Vec<[usize; 4]> {
    
    let mut actions = Vec::with_capacity(264);
    
    for (i,j) in get_towers(&state) {
        for action in get_tower_actions(state,i,j) {
            actions.push(action);
        }
        
    }
    
    actions
    
}

fn get_towers(state: &[[i32; 9]; 9]) -> Vec<(usize,usize)> {
    
    let mut towers = Vec::with_capacity(48);
    
    for i in 0..9 {
        for j in 0..9 {
            if state[i][j] != 0 {
                towers.push((i,j));
            }
        }
    }

    towers

}

fn get_tower_actions(state: &[[i32; 9]; 9],i : usize, j : usize) -> Vec<[usize; 4]>{

    let mut actions = Vec::with_capacity(8);
    
    if state[i][j] == 5 {
        return actions;
    }
    
    for di in 0..3 as i32 {
        for dj in 0..3 as i32{

            let a = i as i32 + di - 1;
            let b = j as i32 + dj - 1;

            if a < 0 || b < 0{
                continue;
            }
                
            let action = [i,j,a.try_into().unwrap(),b.try_into().unwrap()];

            
            if valid_action(state,& action) {
                actions.push(action);
            } else {
            }

        }
    }



    actions

}

fn valid_action(state: &[[i32; 9]; 9], action : &[usize;4]) -> bool {

    let i1 = action[0];
    let j1 = action[1];
    let i2 = action[2];
    let j2 = action[3];


    // no need to check for negative index as it is forced by usize type
    if i1 >= 9 || j1 >= 9 || i2 >= 9 || j2 >= 9 {
        return false
    }
    
    if i1 == i2 && j1 == j2 {
        return false
    } 
    if j1.abs_diff(j2) > 1 ||  i1.abs_diff(i2) > 1 {
        return false
    } 
    
    let h1 = state[i1][j1].abs();
    let h2 = state[i2][j2].abs();
    
    if h1 >= 5 || h2 >= 5 || h1 + h2 > 5 || h1 <= 0 || h2 <= 0{
        return false
    }

    true

}

#[pyfunction]
fn play_action(mut state: [[i32; 9]; 9], action : [usize;4]) -> [[i32; 9]; 9] {

    let h1 = state[action[0]][action[1]];
    let h2 = state[action[2]][action[3]];

    state[action[0]][action[1]] = 0;
    state[action[2]][action[3]] = h1.signum() * (h1.abs() + h2.abs());

    state

}



/// A Python module implemented in Rust.
#[pymodule]
fn rust_opti(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_outcomes, m)?)?;
    m.add_function(wrap_pyfunction!(play_action, m)?)?;
    Ok(())
}
