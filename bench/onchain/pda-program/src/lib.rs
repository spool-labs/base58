//! PDA cost probes on the same stack tape uses (solana-program).
//!
//! Op 0: no-op with the same accounts, the baseline to subtract.
//! Op 1: create the PDA account via invoke_signed, single shot.
//! Op 2: create_program_address loop, address checked against the payload.
//! Op 3: find_program_address loop, address and bump checked.

use core::hint::black_box;
use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    program::invoke_signed,
    program_error::ProgramError,
    pubkey::Pubkey,
};
use solana_system_interface::instruction as system_instruction;

entrypoint!(process_instruction);

fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    data: &[u8],
) -> ProgramResult {
    let op = data[0];
    let iters = data[1] as usize;
    let payload = &data[2..];

    match op {
        0 => {
            black_box((accounts, payload));
        }
        1 => {
            let bump = payload[0];
            let lamports = u64::from_le_bytes(payload[1..9].try_into().unwrap());
            let space = u64::from_le_bytes(payload[9..17].try_into().unwrap());
            let payer = &accounts[0];
            let pda = &accounts[1];
            invoke_signed(
                &system_instruction::create_account(
                    payer.key, pda.key, lamports, space, program_id,
                ),
                accounts,
                &[&[b"vault", payer.key.as_ref(), &[bump]]],
            )?;
        }
        2 => {
            let key = &payload[..32];
            let bump = [payload[32]];
            let expected = &payload[33..65];
            let seeds: [&[u8]; 3] = [b"vault", key, &bump];
            for _ in 0..iters {
                let seeds: &[&[u8]] = black_box(&seeds);
                let derived = Pubkey::create_program_address(seeds, program_id)
                    .map_err(|_| ProgramError::InvalidSeeds)?;
                if derived.as_ref() != expected {
                    return Err(ProgramError::InvalidSeeds);
                }
                black_box(&derived);
            }
        }
        3 => {
            let key = &payload[..32];
            let expected = &payload[32..64];
            let expected_bump = payload[64];
            let seeds: [&[u8]; 2] = [b"vault", key];
            for _ in 0..iters {
                let seeds: &[&[u8]] = black_box(&seeds);
                let (derived, bump) = Pubkey::find_program_address(seeds, program_id);
                if derived.as_ref() != expected || bump != expected_bump {
                    return Err(ProgramError::InvalidSeeds);
                }
                black_box(&derived);
            }
        }
        _ => return Err(ProgramError::InvalidInstructionData),
    }
    Ok(())
}
