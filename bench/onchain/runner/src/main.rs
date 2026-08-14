//! Runs each cu-bench op through Mollusk, subtracts the empty-loop baseline
//! for the same payload, and prints per-call CU.

use mollusk_svm::Mollusk;
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
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

    // --- PDA costs, measured on the solana-program stack ---
    let pda_program_id = Pubkey::new_unique();
    let mut m2 = Mollusk::new(&pda_program_id, "pda_bench");
    m2.compute_budget.compute_unit_limit = 50_000_000;

    let payer = Pubkey::new_unique();
    let (pda, bump) = Pubkey::find_program_address(&[b"vault", payer.as_ref()], &pda_program_id);
    println!("\npda bump {bump} ({} create_program_address tries inside find)", 255 - bump + 1);

    let mut p2 = payer.to_bytes().to_vec();
    p2.push(bump);
    p2.extend_from_slice(pda.as_ref());
    println!("{:<28} {:>8.1} CU/call", "create_program_address", per_call(&m2, pda_program_id, 2, &p2));

    let mut p3 = payer.to_bytes().to_vec();
    p3.extend_from_slice(pda.as_ref());
    p3.push(bump);
    println!("{:<28} {:>8.1} CU/call", "find_program_address", per_call(&m2, pda_program_id, 3, &p3));

    let space = 0u64;
    let lamports = m2.sysvars.rent.minimum_balance(space as usize);
    let (sys_id, sys_acct) = mollusk_svm::program::keyed_account_for_system_program();
    let metas = vec![
        AccountMeta::new(payer, true),
        AccountMeta::new(pda, false),
        AccountMeta::new_readonly(sys_id, false),
    ];
    let accounts = vec![
        (payer, Account { lamports: 1_000_000_000, ..Account::default() }),
        (pda, Account::default()),
        (sys_id, sys_acct),
    ];

    let mut create_data = vec![1u8, 0, bump];
    create_data.extend_from_slice(&lamports.to_le_bytes());
    create_data.extend_from_slice(&space.to_le_bytes());
    let create_ix = Instruction::new_with_bytes(pda_program_id, &create_data, metas.clone());
    let create_res = m2.process_instruction(&create_ix, &accounts);
    assert!(!create_res.program_result.is_err(), "create failed: {:?}", create_res.program_result);
    let created = create_res
        .resulting_accounts
        .iter()
        .find(|(k, _)| *k == pda)
        .expect("pda account in results");
    assert_eq!(created.1.owner, pda_program_id, "pda not owned by program");

    let base_ix = Instruction::new_with_bytes(pda_program_id, &[0u8, 0], metas);
    let base_res = m2.process_instruction(&base_ix, &accounts);
    assert!(!base_res.program_result.is_err());

    println!(
        "{:<28} {:>8} CU  (whole instruction {}, entrypoint baseline {})",
        "create PDA account (CPI)",
        create_res.compute_units_consumed - base_res.compute_units_consumed,
        create_res.compute_units_consumed,
        base_res.compute_units_consumed,
    );
}
