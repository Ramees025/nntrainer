// NNTRAINER_BIN_IO_BEGIN
// ============================================================================
// Binary I/O extension for tokenizers-cpp (added by nntrainer)
//
// This block is appended to mlc-ai/tokenizers-cpp's `rust/src/lib.rs` to add
// fast binary serialization/deserialization based on rmp-serde (MessagePack).
//
// Format:
//   - 9-byte magic:   b"NTRTKBIN\0"
//   - 4-byte LE u32:  format version (currently 1)
//   - rmp-serde encoded `tokenizers::Tokenizer` payload
//
// The magic header allows the C++ side (or future readers) to distinguish this
// file from a raw MessagePack blob and to fall back to JSON when the header is
// missing or the version does not match.
// ============================================================================

const NTR_TKBIN_MAGIC: &[u8; 9] = b"NTRTKBIN\0";
const NTR_TKBIN_VERSION: u32 = 1;
const NTR_TKBIN_HEADER_LEN: usize = 9 + 4;

#[no_mangle]
extern "C" fn tokenizers_new_from_bin(
    input_bin: *const u8,
    len: usize,
) -> *mut TokenizerWrapper {
    if input_bin.is_null() || len < NTR_TKBIN_HEADER_LEN {
        return std::ptr::null_mut();
    }
    unsafe {
        let buf = std::slice::from_raw_parts(input_bin, len);
        if &buf[..NTR_TKBIN_MAGIC.len()] != NTR_TKBIN_MAGIC {
            return std::ptr::null_mut();
        }
        let mut ver_bytes = [0u8; 4];
        ver_bytes.copy_from_slice(&buf[NTR_TKBIN_MAGIC.len()..NTR_TKBIN_HEADER_LEN]);
        if u32::from_le_bytes(ver_bytes) != NTR_TKBIN_VERSION {
            return std::ptr::null_mut();
        }
        let payload = &buf[NTR_TKBIN_HEADER_LEN..];
        match rmp_serde::from_slice::<tokenizers::tokenizer::Tokenizer>(payload) {
            Ok(tokenizer) => Box::into_raw(Box::new(TokenizerWrapper {
                tokenizer,
                decode_str: String::new(),
                id_to_token_result: String::new(),
            })),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_save_to_bin(
    handle: *mut TokenizerWrapper,
    path: *const std::os::raw::c_char,
    path_len: usize,
) -> std::os::raw::c_int {
    if handle.is_null() || path.is_null() || path_len == 0 {
        return -1;
    }
    unsafe {
        let path_bytes = std::slice::from_raw_parts(path as *const u8, path_len);
        let path_str = match std::str::from_utf8(path_bytes) {
            Ok(s) => s,
            Err(_) => return -2,
        };
        let wrapper = &*handle;
        let payload = match rmp_serde::to_vec(&wrapper.tokenizer) {
            Ok(v) => v,
            Err(_) => return -3,
        };
        let mut buf = Vec::with_capacity(NTR_TKBIN_HEADER_LEN + payload.len());
        buf.extend_from_slice(NTR_TKBIN_MAGIC);
        buf.extend_from_slice(&NTR_TKBIN_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);
        let tmp_path = format!("{}.tmp", path_str);
        if std::fs::write(&tmp_path, &buf).is_err() {
            return -4;
        }
        if std::fs::rename(&tmp_path, path_str).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
            return -5;
        }
        0
    }
}
// NNTRAINER_BIN_IO_END
