//! Holds the generated tables to the identities they are supposed to satisfy

/// A non-negative integer as 32-bit words, most significant first
#[derive(Clone, Debug, Eq, PartialEq)]
struct Big(Vec<u32>);

impl Big {
    fn zero() -> Big {
        Big(Vec::new())
    }

    fn from_word(value: u32) -> Big {
        match value {
            0 => Big::zero(),
            _ => Big(vec![value]),
        }
    }

    /// Two raised to this power
    fn two_to(exponent: usize) -> Big {
        let mut words = vec![0u32; exponent / 32 + 1];
        words[exponent / 32] = 1 << (exponent % 32);
        words.reverse();
        Big(words).trimmed()
    }

    fn trimmed(mut self) -> Big {
        while self.0.first() == Some(&0) {
            self.0.remove(0);
        }
        self
    }

    fn multiply(&self, factor: u32) -> Big {
        let mut carry = 0u64;
        let mut words: Vec<u32> = self
            .0
            .iter()
            .rev()
            .map(|word| {
                let product = *word as u64 * factor as u64 + carry;
                carry = product >> 32;
                product as u32
            })
            .collect();
        while carry != 0 {
            words.push(carry as u32);
            carry >>= 32;
        }
        words.reverse();
        Big(words).trimmed()
    }

    fn add(&self, other: &Big) -> Big {
        let mut words = Vec::new();
        let mut carry = 0u64;
        let (left, right) = (self.0.iter().rev(), other.0.iter().rev());
        let width = self.0.len().max(other.0.len());
        let mut left: Vec<u32> = left.copied().collect();
        let mut right: Vec<u32> = right.copied().collect();
        left.resize(width, 0);
        right.resize(width, 0);
        for at in 0..width {
            let sum = left[at] as u64 + right[at] as u64 + carry;
            words.push(sum as u32);
            carry = sum >> 32;
        }
        while carry != 0 {
            words.push(carry as u32);
            carry >>= 32;
        }
        words.reverse();
        Big(words).trimmed()
    }

    /// Raise the limb base to a power
    fn limb_base_to(exponent: usize) -> Big {
        let mut value = Big::from_word(1);
        for _ in 0..exponent {
            for _ in 0..5 {
                value = value.multiply(58);
            }
        }
        value
    }
}

/// Every encode row rebuilds the power of two it stands for
fn check_encode(table: &[&[u32]], words: usize, limbs: usize) {
    for (row, factors) in table.iter().enumerate() {
        let mut total = Big::zero();
        for (column, factor) in factors.iter().enumerate() {
            let place = Big::limb_base_to(limbs - 2 - column);
            total = total.add(&place.multiply(*factor));
        }
        assert_eq!(
            total,
            Big::two_to(32 * (words - 1 - row)),
            "encode row {row}"
        );
    }
}

/// Every decode row rebuilds the power of the limb base it stands for
fn check_decode(table: &[&[u32]], words: usize, limbs: usize) {
    for (row, factors) in table.iter().enumerate() {
        let mut total = Big::zero();
        for (column, factor) in factors.iter().enumerate() {
            let place = Big::two_to(32 * (words - 1 - column));
            total = total.add(&place.multiply(*factor));
        }
        assert_eq!(
            total,
            Big::limb_base_to(limbs - 1 - row),
            "decode row {row}"
        );
    }
}

// the 32-byte encode table rebuilds every power of two it claims
#[test]
fn encode_32() {
    let rows: Vec<&[u32]> = tape_base58::testing::ENCODE_32
        .iter()
        .map(|r| &r[..])
        .collect();
    check_encode(&rows, 8, 9);
}

// the 64-byte encode table rebuilds every power of two it claims
#[test]
fn encode_64() {
    let rows: Vec<&[u32]> = tape_base58::testing::ENCODE_64
        .iter()
        .map(|r| &r[..])
        .collect();
    check_encode(&rows, 16, 18);
}

// the 32-byte decode table rebuilds every power of the limb base it claims
#[test]
fn decode_32() {
    let rows: Vec<&[u32]> = tape_base58::testing::DECODE_32
        .iter()
        .map(|r| &r[..])
        .collect();
    check_decode(&rows, 8, 9);
}

// the 64-byte decode table rebuilds every power of the limb base it claims
#[test]
fn decode_64() {
    let rows: Vec<&[u32]> = tape_base58::testing::DECODE_64
        .iter()
        .map(|r| &r[..])
        .collect();
    check_decode(&rows, 16, 18);
}
