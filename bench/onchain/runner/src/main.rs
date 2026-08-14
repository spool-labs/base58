//! Runs each cu-bench op through Mollusk, subtracts the empty-loop baseline
//! for the same payload, and prints per-call CU.

use mollusk_svm::Mollusk;
use solana_instruction::Instruction;
use solana_pubkey::Pubkey;

const LOW: u8 = 10;
const HIGH: u8 = 60;

fn measure(mollusk: &Mollusk, program_id: Pubkey, op: u8, iters: u8, payload: &[u8]) -> u64 {
    let mut data = vec![op, iters];
    data.extend_from_slice(payload);
    let ix = Instruction::new_with_bytes(program_id, &data, vec![]);
    let result = mollusk.process_instruction(&ix, &[]);
    if !result.program_result.is_err() {
        result.compute_units_consumed
    } else {
        panic!("op {op} failed: {:?}", result.program_result);
    }
}

/// CU per call as the slope between two iteration counts, cancelling
/// entrypoint and setup cost and exposing a hoisted loop as ~0
fn per_call(mollusk: &Mollusk, program_id: Pubkey, op: u8, payload: &[u8]) -> f64 {
    let low = measure(mollusk, program_id, op, LOW, payload);
    let high = measure(mollusk, program_id, op, HIGH, payload);
    (high.saturating_sub(low)) as f64 / (HIGH - LOW) as f64
}

fn main() {
    let program_id = Pubkey::new_unique();
    let mut mollusk = Mollusk::new(&program_id, "cu_bench");
    mollusk.compute_budget.compute_unit_limit = 50_000_000;

    // Deterministic, representative inputs: no leading zeros, mixed bytes
    let key: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
    let sig: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(7));
    let key_enc = bs58::encode(&key).into_string();
    let sig_enc = bs58::encode(&sig).into_string();
    println!("key encodes to {} chars, sig to {}", key_enc.len(), sig_enc.len());

    let cases: [(&str, u8, &[u8]); 12] = [
        ("tape  encode_32", 1, &key),
        ("five8 encode_32", 5, &key),
        ("bs58  encode_32", 9, &key),
        ("tape  decode_32", 2, key_enc.as_bytes()),
        ("five8 decode_32", 6, key_enc.as_bytes()),
        ("bs58  decode_32", 10, key_enc.as_bytes()),
        ("tape  encode_64", 3, &sig),
        ("five8 encode_64", 7, &sig),
        ("bs58  encode_64", 11, &sig),
        ("tape  decode_64", 4, sig_enc.as_bytes()),
        ("five8 decode_64", 8, sig_enc.as_bytes()),
        ("bs58  decode_64", 12, sig_enc.as_bytes()),
    ];

    let mut verify_payload = key.to_vec();
    verify_payload.extend_from_slice(&sig);
    measure(&mollusk, program_id, 13, 200, &verify_payload);
    println!("cross-codec verification passed (200 fuzzed rounds)");

    println!("{:<16} {:>10}", "case", "CU/call");
    for (name, op, payload) in cases {
        let cu = per_call(&mollusk, program_id, op, payload);
        println!("{:<16} {:>10.1}", name, cu);
    }
}
