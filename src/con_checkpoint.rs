//! CON checkpoint the host writes.
//!
//! [`crate::Report`] is a value and a coordinate vector. Symbols, masses,
//! the cell, the fixed mask, and the header line belong to the host.
//! [`write_checkpoint`] uses [`readcon_core::writer::ConFrameWriter::new`]:
//! RoundTrip text, canonical left off, and no decimal precision.
//! [`read_checkpoint`] returns that vector for a later
//! [`crate::minimize`] `init`. [`crate::Solver::step`] does not open a path.
//! [`crate::Report`] gains no path field.
//!
//! CON groups atoms by symbol. Each atom is stored with its index in
//! `report.coords` as `atom_id`. The read places each row back at that
//! index, so a resumed minimization sees the same bits as one that
//! never left memory. Ids that are not a permutation of `0..n-1` are
//! copied in file order.

use std::fs::File;
use std::path::Path;

use ndarray::Array1;
use readcon_core::iterators::read_first_frame;
use readcon_core::types::{AtomDatum, ConFrameBuilder, lengths_and_angles};
use readcon_core::units::default_v3_units_json;
use readcon_core::writer::ConFrameWriter;

use crate::error::{Error, Result};
use crate::report::Report;

/// Write one CON frame for `report`.
///
/// `symbols`, `masses`, and `report.coords` are parallel: the coordinate
/// vector is `3 * symbols.len()` and there is one mass per symbol, in
/// that same order. An empty `fixed` mask leaves every atom free.
/// Otherwise it has one `[x, y, z]` mask per atom. `header` is line 1
/// and must not contain a newline.
///
/// A finite [`Report::value`] is stored with `set_energy`. A non-finite
/// value omits the energy key, because `set_energy` of that value is
/// `NonFiniteValue` on the way back in. Two masses for one symbol are
/// `MassMismatch` from the builder. The frame is built before the path
/// is created, so that refusal does not truncate an existing file.
///
/// # Errors
///
/// Length mismatches, a header that spans lines, and any error the
/// writer or the filesystem returns.
pub fn write_checkpoint<S>(
    path: impl AsRef<Path>,
    report: &Report,
    symbols: &[S],
    masses: &[f64],
    cell: [[f64; 3]; 3],
    fixed: &[[bool; 3]],
    header: &str,
) -> Result<()>
where
    S: AsRef<str>,
{
    let n = symbols.len();
    if report.coords.len() != 3 * n {
        return Err(Error::Checkpoint(format!(
            "coords length {} != 3 * {n} symbols",
            report.coords.len()
        )));
    }
    if masses.len() != n {
        return Err(Error::Checkpoint(format!(
            "masses length {} != {n} symbols",
            masses.len()
        )));
    }
    if !fixed.is_empty() && fixed.len() != n {
        return Err(Error::Checkpoint(format!(
            "fixed mask length {} != {n} symbols",
            fixed.len()
        )));
    }
    if header.contains('\n') || header.contains('\r') {
        return Err(Error::Checkpoint(
            "header must be a single line".to_string(),
        ));
    }

    let (lengths, angles) = lengths_and_angles(cell);
    let mut builder = ConFrameBuilder::new(lengths, angles);
    builder.prebox_header(header);
    builder
        .set_units(default_v3_units_json())
        .map_err(|err| Error::Checkpoint(err.to_string()))?;
    if report.value.is_finite() {
        builder.set_energy(report.value);
    }
    for i in 0..n {
        let mask = if fixed.is_empty() {
            [false; 3]
        } else {
            fixed[i]
        };
        builder.add_atom(
            symbols[i].as_ref(),
            0.0,
            0.0,
            0.0,
            mask,
            i as u64,
            masses[i],
        );
    }
    {
        let dst = builder.positions_mut();
        if let Some(src) = report.coords.as_slice() {
            dst.copy_from_slice(src);
        } else {
            for (dst, src) in dst.iter_mut().zip(report.coords.iter()) {
                *dst = *src;
            }
        }
    }
    let frame = builder
        .build()
        .map_err(|err| Error::Checkpoint(err.to_string()))?;

    let file = File::create(path.as_ref()).map_err(|err| Error::Checkpoint(err.to_string()))?;
    let mut writer = ConFrameWriter::new(file);
    writer
        .write_frame(&frame)
        .map_err(|err| Error::Checkpoint(err.to_string()))?;
    writer
        .flush()
        .map_err(|err| Error::Checkpoint(err.to_string()))?;
    Ok(())
}

/// Positions and energy stored by [`write_checkpoint`].
///
/// The position vector is the frame's f64 slice, one copy, with each
/// row placed at the `atom_id` this writer stored. `energy` is `None`
/// when the key was omitted.
///
/// # Errors
///
/// A missing frame, a position block that is not f64, or a length that
/// is not `3N`.
pub fn read_checkpoint(path: impl AsRef<Path>) -> Result<(Array1<f64>, Option<f64>)> {
    let frame =
        read_first_frame(path.as_ref()).map_err(|err| Error::Checkpoint(err.to_string()))?;
    let slice = frame
        .positions
        .f64_slice()
        .ok_or_else(|| Error::Checkpoint("positions are not f64".to_string()))?;
    let n = frame.atom_data.len();
    if slice.len() != 3 * n {
        return Err(Error::Checkpoint(format!(
            "position length {} != 3 * {n}",
            slice.len()
        )));
    }
    let coords = coords_in_host_order(slice, &frame.atom_data);
    Ok((coords, frame.header.energy()))
}

fn coords_in_host_order(slice: &[f64], atoms: &[AtomDatum]) -> Array1<f64> {
    let mut coords = Array1::zeros(slice.len());
    if host_index_ids(atoms) {
        for (row, atom) in atoms.iter().enumerate() {
            let at = atom.atom_id as usize * 3;
            for k in 0..3 {
                coords[at + k] = slice[row * 3 + k];
            }
        }
    } else {
        for (dst, src) in coords.iter_mut().zip(slice.iter()) {
            *dst = *src;
        }
    }
    coords
}

fn host_index_ids(atoms: &[AtomDatum]) -> bool {
    let n = atoms.len();
    let mut seen = vec![false; n];
    for atom in atoms {
        let Ok(id) = usize::try_from(atom.atom_id) else {
            return false;
        };
        if id >= n || seen[id] {
            return false;
        }
        seen[id] = true;
    }
    true
}
