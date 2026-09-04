use std::{
    borrow::Cow,
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use arc_swap::ArcSwap;
use fjall::{KeyspaceCreateOptions, KvSeparationOptions, PersistMode, Readable};
use monero_oxide::transaction::Transaction;
use rand::Rng;
use tapes::{Persistence, TapeOpenOptions, Tapes, TapesAppend, TapesRead, TapesReadTransaction};

use cuprate_helper::cast::{u32_to_usize, u64_to_usize, usize_to_u64};
use cuprate_pruning::{PruningSeed, CRYPTONOTE_PRUNING_LOG_STRIPES, CRYPTONOTE_PRUNING_TIP_BLOCKS};

use crate::{
    config::Config,
    types::{Amount, BlockInfo, RctOutput, TxInfo},
    BlockchainError,
};

/// The key used to store the main-chain tip in [`BlockchainDatabase::chain_tip`].
pub(crate) const CHAIN_TIP_KEY: &[u8] = b"tip";

/// The amount of times [`BlockchainDatabase::read_transactions`] will retry before giving up.
///
/// With [`TIPS_MATCH_RETRY_DELAY`] this waits 60 seconds, the bound only has to outlast a write, a pair that still disagrees after it is not waiting on one.
const TIPS_MATCH_RETRIES: usize = 6_000;

/// The amount of time [`BlockchainDatabase::read_transactions`] sleeps between retries.
const TIPS_MATCH_RETRY_DELAY: Duration = Duration::from_millis(10);

/// Deletes a [`fjall::Keyspace`] and recreates it with the same name.
fn recreate_fjall_keyspace(
    database: &fjall::Database,
    keyspace: &fjall::Keyspace,
) -> Result<fjall::Keyspace, BlockchainError> {
    let name = keyspace.name().to_string();

    database.delete_keyspace(keyspace.clone())?;
    Ok(database.keyspace(&name, KeyspaceCreateOptions::default)?)
}

/// Deletes a [`fjall::Keyspace`] and recreates it with the same name.
pub(crate) fn reset_fjall_keyspace(
    database: &fjall::Database,
    keyspace: &ArcSwap<fjall::Keyspace>,
) -> Result<(), BlockchainError> {
    let new_keyspace = recreate_fjall_keyspace(database, &keyspace.load())?;
    keyspace.store(Arc::new(new_keyspace));

    Ok(())
}

/// The [`KeyspaceCreateOptions`] for [`BlockchainDatabase::prunable_tip`].
///
/// It is created in 2 places, [`BlockchainDatabase::open_with_fjall_database`] and [`BlockchainDatabase::enable_pruning`], and both must use the same options.
fn prunable_tip_options() -> KeyspaceCreateOptions {
    KeyspaceCreateOptions::default().with_kv_separation(Some(
        KvSeparationOptions::default()
            .separation_threshold(3_000)
            .compression(fjall::CompressionType::None),
    ))
}

/// The [`TapeOpenOptions`] for a tape in `dir`, keeping `top_cache_size` bytes of its top in memory.
fn tape_options(top_cache_size: u64, dir: &Path) -> TapeOpenOptions {
    TapeOpenOptions {
        top_cache_size,
        dir: dir.to_path_buf(),
    }
}

/// Reads the [`PruningSeed`] committed to the `tapes_metadata` tape.
///
/// [`BlockchainDatabase::enable_pruning`] is the only writer of that tape, so an empty one means this node was never pruned.
///
/// # Panics
///
/// This will panic if the committed seed is not a valid [`PruningSeed`], only a corrupt tape can hold one.
fn read_pruning_seed(
    tapes: &impl TapesRead,
    tapes_metadata: &tapes::BlobTape,
) -> Result<PruningSeed, BlockchainError> {
    if tapes.blob_tape_len(tapes_metadata).unwrap_or(0) == 0 {
        return Ok(PruningSeed::NotPruned);
    }

    let mut seed_bytes = [0; 4];
    tapes.read_bytes(tapes_metadata, 0, &mut seed_bytes)?;

    Ok(PruningSeed::decompress(u32::from_le_bytes(seed_bytes)).unwrap())
}

/// Opens the [`BlockchainDatabase::prunable_blobs`] tapes `pruning_seed` tells us to keep, leaving the rest as [`None`].
///
/// Open every tape while the seed is not committed, [`BlockchainDatabase::enable_pruning`] fills `prunable_tip` from all of them.
/// Once it is committed only the stripe we keep is opened, the other 7 are deleted by the caller.
fn open_prunable_tapes(
    tape_append_tx: &mut tapes::TapesAppendTransaction,
    pruning_seed: PruningSeed,
    options: &TapeOpenOptions,
) -> Result<Vec<Option<tapes::BlobTape>>, BlockchainError> {
    let prunable_blobs = (0..8)
        .map(|i| {
            if pruning_seed
                .get_stripe()
                .is_none_or(|stripe| u32_to_usize(stripe) - 1 == i)
            {
                tape_append_tx
                    .open_blob_tape(PRUNABLE_BLOBS[i], options)
                    .map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(prunable_blobs)
}

/// The blockchain database.
pub struct BlockchainDatabase {
    /// The database configuration.
    pub(crate) config: Config,

    /// The tapes database.
    pub(crate) linear_tapes: Tapes,
    /// The fjall database.
    pub(crate) fjall: fjall::Database,

    /// Block heights:
    ///
    /// | key                  | value                               |
    /// |----------------------|-------------------------------------|
    /// | block hash: [u8; 32] | block height: usize (little endian) |
    pub(crate) block_heights: fjall::Keyspace,
    /// Main-chain tip:
    ///
    /// | key               | value                |
    /// |-------------------|----------------------|
    /// | [`CHAIN_TIP_KEY`] | block hash: [u8; 32] |
    pub(crate) chain_tip: fjall::Keyspace,
    /// Key images:
    ///
    /// | key                 | value |
    /// |---------------------|-------|
    /// | key image: [u8; 32] | []    |
    pub(crate) key_images: fjall::Keyspace,
    /// Pre-RCT outputs:
    ///
    /// | key                                     | value                             |
    /// |-----------------------------------------|-----------------------------------|
    /// | The ID of the output [`PreRctOutputId`] | The output data: [`Output`] bytes |
    pub(crate) pre_rct_outputs: fjall::Keyspace,
    /// Transaction IDs:
    ///
    /// | key               | value                      |
    /// |-------------------|----------------------------|
    /// | Tx hash: [u8; 32] | Tx ID: u64 (little endian) |
    pub(crate) tx_ids: fjall::Keyspace,
    /// V1 transaction output amount indices:
    ///
    /// | key                        | value                                           |
    /// |----------------------------|--------------------------------------------------|
    /// | Tx ID: u64 (little endian) | amount indices as a [u64] (little endian) slice |
    pub(crate) v1_tx_outputs: fjall::Keyspace,
    /// Alt chain info:
    ///
    /// | key                           | value                  |
    /// |-------------------------------|------------------------|
    /// | Chain ID: u64 (little endian) | [`AltChainInfo`] bytes |
    pub(crate) alt_chain_infos: ArcSwap<fjall::Keyspace>,
    /// Alt block heights:
    ///
    /// | key                  | value                    |
    /// |----------------------|--------------------------|
    /// | block hash: [u8; 32] | [`AltBlockHeight`] bytes |
    pub(crate) alt_block_heights: ArcSwap<fjall::Keyspace>,
    /// Alt block info:
    ///
    /// | key                        | value                          |
    /// |----------------------------|--------------------------------|
    /// | [`AltBlockHeight`] bytes   | [`CompactAltBlockInfo`] bytes  |
    pub(crate) alt_block_infos: ArcSwap<fjall::Keyspace>,
    /// Alt block blobs:
    ///
    /// | key                      | value            |
    /// |--------------------------|------------------|
    /// | [`AltBlockHeight`] bytes | block blob: [u8] |
    pub(crate) alt_block_blobs: ArcSwap<fjall::Keyspace>,
    /// Alt transaction blobs:
    ///
    /// | key                        | value                       |
    /// |----------------------------|-----------------------------|
    /// | transaction hash: [u8; 32] | full transaction blob: [u8] |
    pub(crate) alt_transaction_blobs: ArcSwap<fjall::Keyspace>,
    /// Alt transaction info:
    ///
    /// | key                        | value                        |
    /// |----------------------------|------------------------------|
    /// | transaction hash: [u8; 32] | [`AltTransactionInfo`] bytes |
    pub(crate) alt_transaction_infos: ArcSwap<fjall::Keyspace>,

    /// RCT (v2+) outputs, indexed sequentially.
    ///
    /// | index                 | value         |
    /// |-----------------------|---------------|
    /// | RCT output index: u64 | [`RctOutput`] |
    pub(crate) rct_outputs: tapes::FixedSizedTape<RctOutput>,
    /// Transaction info, indexed by [`TxId`].
    ///
    /// | index      | value      |
    /// |------------|------------|
    /// | Tx ID: u64 | [`TxInfo`] |
    pub(crate) tx_infos: tapes::FixedSizedTape<TxInfo>,
    /// Block info, indexed by block height.
    ///
    /// | index             | value         |
    /// |-------------------|---------------|
    /// | Block height: u64 | [`BlockInfo`] |
    pub(crate) block_infos: tapes::FixedSizedTape<BlockInfo>,
    /// Pruned blobs.
    ///
    /// The format for this blob-tape per each block is:
    ///
    /// | data                                       |
    /// |--------------------------------------------|
    /// | block blob (header, miner tx, tx hashes)   |
    /// | tx 0 pruned blob                           |
    /// | tx 0 prunable hash (32 bytes)              |
    /// | tx 1 pruned blob                           |
    /// | tx 1 prunable hash (32 bytes)              |
    /// | ...                                        |
    ///
    /// The prunable hash is `[0; 32]` for v1 txs.
    /// Each block is appended directly after the one before it.
    pub(crate) pruned_blobs: tapes::BlobTape,
    /// V1 prunable transaction blobs, indexed by [`TxInfo::prunable_blob_idx`].
    ///
    /// This tape stores the prunable blob for all V1 txs, these can't be pruned.
    pub(crate) v1_prunable_blobs: tapes::BlobTape,
    /// V2+ prunable transaction blobs, split across 8 stripes.
    /// Indexed by [`TxInfo::prunable_blob_idx`].
    ///
    /// These tapes store the prunable part of each tx, the stripe a tx is stored in depends on the
    /// height of the block.
    ///
    /// Each blob tape is stored in an [`Option`] to allow for pruning.
    pub(crate) prunable_blobs: Vec<Option<tapes::BlobTape>>,

    /// Metadata of the database, currently stores the pruning seed.
    pub(crate) tapes_metadata: tapes::BlobTape,

    /// Includes up to the top 5500 blocks prunable blobs, since pruned nodes should keep this.
    ///
    /// In some circumstances this could hold less than that amount of blocks.
    ///
    /// | key                        | value                  |
    /// |----------------------------|------------------------|
    /// | Tx ID: u64 (little endian) | prunable blob: [u8]    |
    pub(crate) prunable_tip: Option<fjall::Keyspace>,

    /// A runtime cache of the number of outputs for each pre-rct output amount.
    /// This is filled in lazily.
    pub(crate) pre_rct_numb_outputs_cache: Mutex<HashMap<Amount, u64>>,

    /// The [`PruningSeed`] for this database.
    pruning_seed: PruningSeed,
}

const PRUNABLE_BLOBS: [&str; 8] = [
    "prunable1",
    "prunable2",
    "prunable3",
    "prunable4",
    "prunable5",
    "prunable6",
    "prunable7",
    "prunable8",
];

impl BlockchainDatabase {
    /// Open a [`BlockchainDatabase`] with an [`fjall::Database`] for storing data that can't be stored in tapes.
    ///
    /// This only opens what is already on disk, the [`PruningSeed`] committed to the tapes decides which `prunable_blobs` tapes come with it.
    /// It does not check that the 2 halves agree, nor does it act on [`Config::prune`], [`BlockchainDatabase::make_consistent`] does both.
    pub fn open_with_fjall_database(
        config: &Config,
        fjall: fjall::Database,
    ) -> Result<Self, BlockchainError> {
        // Everything in here is derived from the tapes, so it can always be rebuilt
        let (block_heights, chain_tip, key_images, pre_rct_outputs, tx_ids, v1_tx_outputs) =
            Self::open_main_chain_keyspaces(&fjall)?;

        // Alt blocks never reach the tapes, so this is the only copy of them
        let (
            alt_chain_infos,
            alt_block_heights,
            alt_block_infos,
            alt_block_blobs,
            alt_transaction_blobs,
            alt_transaction_infos,
        ) = Self::open_alt_chain_keyspaces(&fjall)?;

        // If we already have a `prunable_tip` keyspace then open it here, otherwise we will make a
        // new one and fill it in later if pruning.
        let prunable_tip = fjall
            .keyspace_exists("prunable_tip")
            .then(|| fjall.keyspace("prunable_tip", prunable_tip_options))
            .transpose()?;

        // The tapes => the authoritative copy of the chain.
        let tapes_index_dir = config.index_dir.join("tapes");
        let tapes_blob_dir = config.blob_dir.join("tapes");

        let mut linear_tapes = Tapes::open(&tapes_index_dir)?;
        let mut tape_append_tx = linear_tapes.append();

        // Open every tape, each keeping the amount of its top the config asks for in memory.
        // The `prunable_blobs` ones come last because which of them we open depends on the seed committed to `tapes_metadata`
        let rct_outputs = tape_append_tx.open_fixed_sized_tape(
            "rct_outputs",
            &tape_options(config.cache_sizes.rct_outputs, &tapes_index_dir),
        )?;
        let tx_infos = tape_append_tx.open_fixed_sized_tape(
            "tx_infos",
            &tape_options(config.cache_sizes.tx_infos, &tapes_index_dir),
        )?;
        let block_infos = tape_append_tx.open_fixed_sized_tape(
            "block_infos",
            &tape_options(config.cache_sizes.block_infos, &tapes_index_dir),
        )?;
        let tapes_metadata =
            tape_append_tx.open_blob_tape("tapes_metadata", &tape_options(8, &tapes_index_dir))?;
        let pruned_blobs = tape_append_tx.open_blob_tape(
            "pruned_blobs",
            &tape_options(config.cache_sizes.pruned_blobs, &tapes_blob_dir),
        )?;
        let v1_prunable_blobs = tape_append_tx.open_blob_tape(
            "v1_prunable_blobs",
            &tape_options(config.cache_sizes.v1_prunable_blobs, &tapes_blob_dir),
        )?;

        let prunable_tape_open_options =
            tape_options(config.cache_sizes.prunable_blobs, &tapes_blob_dir);

        let pruning_seed = read_pruning_seed(&tape_append_tx, &tapes_metadata)?;
        let prunable_blobs = open_prunable_tapes(
            &mut tape_append_tx,
            pruning_seed,
            &prunable_tape_open_options,
        )?;

        // Commit before deleting anything.
        // `delete_tape` refuses to run while an append transaction is alive.
        tape_append_tx.commit(Persistence::SyncAll)?;

        // Delete every tape `open_prunable_tapes` left closed.
        // `enable_pruning` already deletes them, this is a no-op unless it was interrupted before it got there
        for (i, prunable_blob) in prunable_blobs.iter().enumerate() {
            if prunable_blob.is_none() {
                linear_tapes.delete_tape(PRUNABLE_BLOBS[i], &prunable_tape_open_options)?;
            }
        }

        tracing::debug!("opened db");
        Ok(Self {
            fjall,
            linear_tapes,
            config: config.clone(),
            block_heights,
            chain_tip,
            key_images,
            pre_rct_outputs,
            tx_ids,
            v1_tx_outputs,
            tapes_metadata,
            alt_chain_infos: ArcSwap::from_pointee(alt_chain_infos),
            alt_block_heights: ArcSwap::from_pointee(alt_block_heights),
            alt_block_infos: ArcSwap::from_pointee(alt_block_infos),
            alt_block_blobs: ArcSwap::from_pointee(alt_block_blobs),
            alt_transaction_blobs: ArcSwap::from_pointee(alt_transaction_blobs),
            alt_transaction_infos: ArcSwap::from_pointee(alt_transaction_infos),
            rct_outputs,
            tx_infos,
            block_infos,
            pruned_blobs,
            v1_prunable_blobs,
            prunable_blobs,
            prunable_tip,
            pre_rct_numb_outputs_cache: Mutex::new(HashMap::new()),
            pruning_seed,
        })
    }

    /// Opens the [`fjall::Keyspace`]s holding main-chain data.
    ///
    /// Every one of these is derived from the tapes.
    /// [`BlockchainDatabase::rebuild_fjall_database`] recreates them by replaying the tapes, so losing any of them is recoverable.
    fn open_main_chain_keyspaces(
        fjall: &fjall::Database,
    ) -> Result<
        (
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
        ),
        BlockchainError,
    > {
        let block_heights = fjall.keyspace("block_heights", KeyspaceCreateOptions::default)?;
        let chain_tip = fjall.keyspace("chain_tip", KeyspaceCreateOptions::default)?;
        let key_images = fjall.keyspace("key_images", KeyspaceCreateOptions::default)?;
        let pre_rct_outputs = fjall.keyspace("pre_rct_outputs", KeyspaceCreateOptions::default)?;
        let tx_ids = fjall.keyspace("tx_ids", KeyspaceCreateOptions::default)?;
        let v1_tx_outputs = fjall.keyspace("tx_outputs", KeyspaceCreateOptions::default)?;

        Ok((
            block_heights,
            chain_tip,
            key_images,
            pre_rct_outputs,
            tx_ids,
            v1_tx_outputs,
        ))
    }

    /// Opens the [`fjall::Keyspace`]s holding alt-chain data.
    ///
    /// These hold the only copy of their data, alt blocks never reach the tapes, so [`BlockchainDatabase::rebuild_fjall_database`] drops them instead of replaying them.
    fn open_alt_chain_keyspaces(
        fjall: &fjall::Database,
    ) -> Result<
        (
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
            fjall::Keyspace,
        ),
        BlockchainError,
    > {
        let alt_chain_infos = fjall.keyspace("alt_chain_infos", KeyspaceCreateOptions::default)?;
        let alt_block_heights =
            fjall.keyspace("alt_block_heights", KeyspaceCreateOptions::default)?;
        let alt_block_infos = fjall.keyspace("alt_block_infos", KeyspaceCreateOptions::default)?;
        let alt_block_blobs = fjall.keyspace("alt_block_blobs", KeyspaceCreateOptions::default)?;
        let alt_transaction_blobs =
            fjall.keyspace("alt_transaction_blobs", KeyspaceCreateOptions::default)?;
        let alt_transaction_infos =
            fjall.keyspace("alt_transaction_infos", KeyspaceCreateOptions::default)?;

        Ok((
            alt_chain_infos,
            alt_block_heights,
            alt_block_infos,
            alt_block_blobs,
            alt_transaction_blobs,
            alt_transaction_infos,
        ))
    }

    /// Returns whether Fjall and Tapes are at the same main-chain tip.
    fn tips_match(
        &self,
        fjall: &impl Readable,
        tapes: &impl TapesRead,
    ) -> Result<bool, BlockchainError> {
        let tapes_height = tapes
            .fixed_sized_tape_len(&self.block_infos)
            .expect("block_infos tape exists");
        let tapes_tip = match tapes_height.checked_sub(1) {
            Some(top_height) => Some(
                tapes
                    .read_entry(&self.block_infos, top_height)?
                    .ok_or(BlockchainError::NotFound)?
                    .block_hash,
            ),
            None => None,
        };
        let fjall_tip = fjall.get(&self.chain_tip, CHAIN_TIP_KEY)?;

        Ok(match (tapes_tip, fjall_tip.as_deref()) {
            (None, None) => true,
            (Some(tapes_tip), Some(fjall_tip)) => tapes_tip.as_slice() == fjall_tip,
            _ => false,
        })
    }

    /// Returns Fjall and Tapes read transactions at the same main-chain tip.
    ///
    /// A write commits the tapes before fjall, so a reader that lands between the 2 commits gets a mismatched pair.
    /// Retrying gives the writer the time it needs to finish.
    ///
    /// # Panics
    ///
    /// This will panic if the tips still disagree after `TIPS_MATCH_RETRIES` retries.
    pub fn read_transactions(
        &self,
    ) -> Result<(fjall::Snapshot, TapesReadTransaction), BlockchainError> {
        for _ in 0..TIPS_MATCH_RETRIES {
            let fjall = self.fjall.snapshot();
            let tapes = self.linear_tapes.reader();

            if self.tips_match(&fjall, &tapes)? {
                return Ok((fjall, tapes));
            }

            std::thread::sleep(TIPS_MATCH_RETRY_DELAY);
        }

        // Fjall and the tapes disagree *and* nothing is closing the gap, so give up
        // Restarting rebuilds fjall from the tapes, spinning here forever would just hang every reader instead
        panic!("fjall and the tapes did not agree on a main-chain tip after {TIPS_MATCH_RETRIES} retries, a write failed part way through, restart to rebuild fjall from the tapes");
    }

    /// Checks if the fjall and tapes database are in sync and rebuilds the fjall database if it
    /// is not.
    pub fn make_consistent(&mut self) -> Result<(), BlockchainError> {
        tracing::info!("Checking blockchain database consistency.");
        let tips_match = {
            let fjall = self.fjall.snapshot();
            let tapes = self.linear_tapes.reader();
            self.tips_match(&fjall, &tapes)?
        };

        if !tips_match {
            tracing::warn!("fjall and tapes are out of sync");
            self.rebuild_fjall_database()?;
        }

        // We are pruned *but* `prunable_tip` is gone, so fail loudly
        if self.pruning_seed != PruningSeed::NotPruned && self.prunable_tip.is_none() {
            return Err(BlockchainError::MissingPrunableTip);
        }

        // Pruning was requested *and* no seed is committed yet, so activate it.
        //
        // This doesn't test `prunable_tip` because a crash mid-activation could leave a half-filled one behind, and skipping on that would leave the node unpruned.
        // Resuming over a half-filled `prunable_tip` would be safe because no seed means no tape was deleted yet, so we rewrite its entries from the same tapes
        if self.config.prune && self.pruning_seed == PruningSeed::NotPruned {
            self.enable_pruning()?;
        }

        Ok(())
    }

    /// Rebuilds the fjall database.
    ///
    /// This will not fill in the prunable tip blocks.
    pub fn rebuild_fjall_database(&mut self) -> Result<(), BlockchainError> {
        self.block_heights = recreate_fjall_keyspace(&self.fjall, &self.block_heights)?;
        self.chain_tip = recreate_fjall_keyspace(&self.fjall, &self.chain_tip)?;
        self.key_images = recreate_fjall_keyspace(&self.fjall, &self.key_images)?;
        self.pre_rct_outputs = recreate_fjall_keyspace(&self.fjall, &self.pre_rct_outputs)?;
        self.tx_ids = recreate_fjall_keyspace(&self.fjall, &self.tx_ids)?;
        self.v1_tx_outputs = recreate_fjall_keyspace(&self.fjall, &self.v1_tx_outputs)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_chain_infos)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_block_heights)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_block_infos)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_block_blobs)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_transaction_blobs)?;
        reset_fjall_keyspace(&self.fjall, &self.alt_transaction_infos)?;

        // It is taken out of `self` for the duration of the replay rather than deleted,
        // because leaving it in place would overwrite every V2 entry with the empty prunable blob the rebuild feeds it.
        // It has to go back even if the replay fails, otherwise `self` would be left believing a pruned node has no tip data at all.
        let prunable_tip = self.prunable_tip.take();
        let res = self.replay_tapes_into_fjall();
        self.prunable_tip = prunable_tip;

        res
    }

    /// Replays the tapes into the fjall keyspaces [`BlockchainDatabase::rebuild_fjall_database`] just recreated.
    fn replay_tapes_into_fjall(&self) -> Result<(), BlockchainError> {
        let rebuild_span = tracing::info_span!("rebuild_fjall_database");
        let _guard = rebuild_span.enter();

        tracing::info!("rebuilding fjall db");

        let tapes_reader = self.linear_tapes.reader();

        let tx_infos_iter = tapes_reader.iter_from(&self.tx_infos, 0)?;
        let mut tx_iter = tx_infos_iter.map(|tx_info| {
            let tx_info = tx_info.unwrap();

            let mut tx_blob = vec![0; tx_info.pruned_size];
            tapes_reader
                .read_bytes(&self.pruned_blobs, tx_info.pruned_blob_idx, &mut tx_blob)
                .unwrap();

            let tx = Transaction::read(&mut tx_blob.as_slice()).unwrap();

            // The prunable blob only ever reaches `prunable_tip`, which is out of `self` for the replay, so an empty one is enough
            (Cow::Owned(tx), Cow::Owned(vec![]))
        });

        let mut batch = self.fjall.batch().durability(Some(PersistMode::Buffer));
        let mut numb_txs = 0;
        for height in 0..tapes_reader
            .fixed_sized_tape_len(&self.block_infos)
            .expect("block_infos tape exists")
        {
            let block =
                crate::ops::block::get_block(&u64_to_usize(height), None, &tapes_reader, self)?;

            // `add_block_to_dynamic_tables` takes the miner tx from the block, drop the tape's copy so `tx_iter` lines up with `block.transactions`
            let _miner_tx = tx_iter.next();

            crate::ops::block::add_block_to_dynamic_tables(
                self,
                &block,
                &block.hash(),
                &mut tx_iter,
                &mut numb_txs,
                &mut batch,
                &mut self.pre_rct_numb_outputs_cache.lock().unwrap(),
            )?;

            if height % 1000 == 0 {
                tracing::info!("{} blocks processed", height);
                let old_batch = std::mem::replace(
                    &mut batch,
                    self.fjall.batch().durability(Some(PersistMode::Buffer)),
                );

                old_batch.commit()?;
            }
        }

        batch.commit()?;

        Ok(())
    }

    /// Returns the [`PruningSeed`] for this database.
    #[inline]
    pub const fn pruning_seed(&self) -> PruningSeed {
        self.pruning_seed
    }

    /// Enables pruning by filling [`BlockchainDatabase::prunable_tip`] from all tapes, then delete the 7 we no longer need.
    ///
    /// **The order matters!, steps 4 and 5 are destructive**:
    ///
    /// 1. Get the [`PruningSeed`] to prune with
    ///     - In memory only, nothing is written yet
    /// 2. Populate `prunable_tip` with the latest blocks
    /// 3. Persist it to disk
    ///     - Committing a `fjall` batch is not enough, it can still be lost on a crash
    /// 4. Commit the [`PruningSeed`]
    ///     - The point of no return, from here on the node counts as pruned
    ///     - The next startup deletes the tapes even if we crash before step 5
    /// 5. Delete the unnecessary [`BlockchainDatabase::prunable_blobs`]
    ///
    /// If interrupted before 4, the node is still unpruned with every tape intact and the next startup redoes this, reusing the partial `prunable_tip` (as long as pruning is still requested)
    ///
    /// # Panics
    ///
    /// This will panic if the node already has a committed [`PruningSeed`], that is what guarantees every prunable tape is open.
    fn enable_pruning(&mut self) -> Result<(), BlockchainError> {
        debug_assert_eq!(self.pruning_seed, PruningSeed::NotPruned);

        // 1. Get the seed to prune with (in memory only)
        // Only a node that is not pruned yet gets here, so the stripe is always a fresh random one
        let stripe_idx = rand::thread_rng().gen_range(
            1..=u32::try_from(PRUNABLE_BLOBS.len())
                .expect("there shouldn't be that many prunable blobs"),
        );
        let seed = PruningSeed::new_pruned(stripe_idx, CRYPTONOTE_PRUNING_LOG_STRIPES).unwrap();

        let stripe = seed.get_stripe().unwrap();

        tracing::info!("Pruning chain on stripe = {stripe:?}.");

        // 2. Populate `prunable_tip`
        let prunable_tip = self.fill_prunable_tip()?;

        // 3. Persist `prunable_tip` to disk
        // Make the tip durable before recording that we are pruned and before deleting the tapes it replaces
        self.fjall.persist(PersistMode::SyncAll)?;

        // 4. Commit the seed (the point of no return)
        let mut tapes_tx = self.linear_tapes.append();
        tapes_tx.append_bytes(&self.tapes_metadata, &seed.compress().to_le_bytes())?;
        tapes_tx.commit(Persistence::SyncAll)?;

        self.pruning_seed = seed;
        self.prunable_tip = Some(prunable_tip);

        // 5. Delete the tapes we no longer need
        self.delete_unnecessary_tapes(stripe)
    }

    /// Fills a `prunable_tip` keyspace with the prunable blobs of the last [`CRYPTONOTE_PRUNING_TIP_BLOCKS`] blocks.
    ///
    /// This is step 2 of [`BlockchainDatabase::enable_pruning`], it writes nothing else and deletes nothing.
    ///
    /// # Panics
    ///
    /// This will panic if the node already has a committed [`PruningSeed`], that is what guarantees every prunable tape is open.
    fn fill_prunable_tip(&self) -> Result<fjall::Keyspace, BlockchainError> {
        // An interrupted run leaves a correct but incomplete keyspace, so reuse it if it is there
        let prunable_tip = self.fjall.keyspace("prunable_tip", prunable_tip_options)?;

        let tapes_reader = self.linear_tapes.reader();
        let mut w = self.fjall.batch();

        let start_tip_height = tapes_reader
            .fixed_sized_tape_len(&self.block_infos)
            .unwrap_or(0)
            .saturating_sub(usize_to_u64(CRYPTONOTE_PRUNING_TIP_BLOCKS));
        let start_tx_idx = tapes_reader
            .read_entry(&self.block_infos, start_tip_height)?
            .map_or(0, |info| info.mining_tx_index);
        let end_tx_idx = tapes_reader
            .fixed_sized_tape_len(&self.tx_infos)
            .unwrap_or(0);

        // We work backwards from the top, if this is stopped part way the cache will be short, which
        // is acceptable.
        for (i, tx_id) in (start_tx_idx..end_tx_idx).rev().enumerate() {
            let tx_info = tapes_reader.read_entry(&self.tx_infos, tx_id)?.unwrap();

            if tx_info.is_v1_tx() {
                continue;
            }
            let tx_stripe = cuprate_pruning::get_block_pruning_stripe(
                tx_info.height,
                usize::MAX,
                CRYPTONOTE_PRUNING_LOG_STRIPES,
            )
            .unwrap();

            // Pruning is only enabled while the seed is not committed
            let prunable_blob = self.prunable_blobs
                [usize::try_from(tx_stripe).expect("stripe will not exceed usize::MAX") - 1]
                .as_ref()
                .expect("every prunable tape is open when enabling pruning");

            let mut blob = vec![0; tx_info.prunable_size];
            tapes_reader.read_bytes(prunable_blob, tx_info.prunable_blob_idx, &mut blob)?;

            w.insert(&prunable_tip, tx_id.to_le_bytes(), blob.as_slice());

            if (i + 1) % 1000 == 0 {
                w.commit()?;
                w = self.fjall.batch();
            }
        }
        w.commit()?;

        Ok(prunable_tip)
    }

    /// Deletes every [`BlockchainDatabase::prunable_blobs`] tape outside `stripe`.
    ///
    /// This is step 5 of [`BlockchainDatabase::enable_pruning`], it must not run before `prunable_tip` is on disk and the [`PruningSeed`] is committed.
    /// TODO: make the tapes delete API better so we don't need to reconstruct this.
    fn delete_unnecessary_tapes(&mut self, stripe: u32) -> Result<(), BlockchainError> {
        let prunable_tape_open_options = tape_options(
            self.config.cache_sizes.prunable_blobs,
            &self.config.blob_dir.join("tapes"),
        );

        for (i, prunable_blob) in self.prunable_blobs.iter_mut().enumerate() {
            if u32_to_usize(stripe) - 1 != i {
                self.linear_tapes
                    .delete_tape(PRUNABLE_BLOBS[i], &prunable_tape_open_options)?;
                *prunable_blob = None;
            }
        }

        Ok(())
    }
}

impl Drop for BlockchainDatabase {
    fn drop(&mut self) {
        tracing::info!(parent: &tracing::Span::none(), "Syncing blockchain database to storage.");

        let _ = self.fjall.persist(PersistMode::SyncAll);

        let _ = self.linear_tapes.append().commit(Persistence::SyncAll);
    }
}
