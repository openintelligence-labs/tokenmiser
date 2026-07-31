//! SSE parsing and replay for the streaming path.

use bytes::Bytes;
use serde_json::{json, Value};
use tokenmiser_providers::{ChatChoice, ChatMessage, ChatResponse, Usage};

/// Cap on the partial-event buffer; an upstream that never emits a blank line
/// would otherwise grow it for the life of the stream.
const MAX_EVENT_BUF: usize = 1024 * 1024;

/// Cap on reassembled content. Past this, accumulation stops and the stream
/// becomes uncacheable; the client still receives every byte.
const MAX_CONTENT: usize = 8 * 1024 * 1024;

/// Incremental parser over an upstream SSE chunk stream, reassembling
/// choice 0 into a cacheable response.
#[derive(Default)]
pub struct StreamAccumulator {
    buf: Vec<u8>,
    overflowed: bool,
    /// The previous packet ended on a blank line whose final terminator was
    /// a bare `\r`; if the next packet starts with the `\n` completing that
    /// CRLF pair, swallow it instead of reading a second terminator.
    swallow_leading_lf: bool,
    saw_error: bool,
    pub content: String,
    pub id: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub saw_done: bool,
    pub multi_choice: bool,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one raw upstream packet; a trailing partial event stays buffered.
    pub fn push(&mut self, packet: &[u8]) {
        if self.overflowed {
            return;
        }
        let mut packet = packet;
        if std::mem::take(&mut self.swallow_leading_lf)
            && self.buf.is_empty()
            && packet.first() == Some(&b'\n')
        {
            packet = &packet[1..];
        }
        self.buf.extend_from_slice(packet);
        let mut last_event_ended_cr = false;
        while let Some(end) = find_event_end(&self.buf) {
            let event: Vec<u8> = self.buf.drain(..end).collect();
            last_event_ended_cr = event.last() == Some(&b'\r');
            if let Ok(text) = std::str::from_utf8(&event) {
                self.parse_event(text);
            }
        }
        self.swallow_leading_lf = last_event_ended_cr && self.buf.is_empty();
        if self.buf.len() > MAX_EVENT_BUF {
            self.overflowed = true;
            self.buf = Vec::new();
        }
    }

    /// True while the forwarded bytes end inside an unterminated SSE event.
    ///
    /// The accumulator sees exactly the bytes sent downstream, so this is the
    /// oracle for suppressing keepalive comment injection: a comment spliced
    /// into the middle of a `data:` line corrupts the client's parse. After an
    /// overflow the boundary is unknown and this reports false, so disconnect
    /// probing keeps working on non-SSE-shaped upstream bytes.
    pub fn is_mid_event(&self) -> bool {
        !self.buf.is_empty()
    }

    fn parse_event(&mut self, event: &str) {
        // Split on every SSE-legal line terminator (LF, CRLF, CR).
        for line in event.split(['\r', '\n']) {
            let payload = match line.strip_prefix("data:") {
                Some(p) => p.trim(),
                None => continue,
            };
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                self.saw_done = true;
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(payload) else {
                continue;
            };
            // An in-band error on a 200 stream means the content so far is
            // truncated; never cache it.
            if v.get("error").is_some_and(|e| !e.is_null()) {
                self.saw_error = true;
            }
            if self.id.is_none() {
                self.id = v.get("id").and_then(|x| x.as_str()).map(String::from);
            }
            if self.created.is_none() {
                self.created = v.get("created").and_then(|x| x.as_u64());
            }
            if self.model.is_none() {
                self.model = v.get("model").and_then(|x| x.as_str()).map(String::from);
            }
            if let Some(u) = v
                .get("usage")
                .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok())
            {
                self.usage = Some(u);
            }
            if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                if choices.len() > 1 {
                    self.multi_choice = true;
                }
                if let Some(c0) = choices.first() {
                    if let Some(fr) = c0.get("finish_reason").and_then(|f| f.as_str()) {
                        self.finish_reason = Some(fr.to_string());
                    }
                    if let Some(delta) = c0.get("delta") {
                        if let Some(s) = delta.get("content").and_then(|c| c.as_str()) {
                            if self.content.len() + s.len() > MAX_CONTENT {
                                self.overflowed = true;
                                self.content = String::new();
                            } else {
                                self.content.push_str(s);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Assemble the accumulated stream into a cacheable `ChatResponse`, or
    /// `None` when it is empty, multi-choice, or never finished cleanly.
    pub fn into_chat_response(self, fallback_model: &str) -> Option<ChatResponse> {
        if self.overflowed || self.saw_error || self.multi_choice || self.content.trim().is_empty()
        {
            return None;
        }
        if !self.saw_done && self.finish_reason.is_none() {
            return None;
        }
        Some(ChatResponse {
            id: self
                .id
                .unwrap_or_else(|| format!("chatcmpl-tm-{}", now_unix())),
            object: "chat.completion".into(),
            created: self.created.unwrap_or_else(now_unix),
            model: self.model.unwrap_or_else(|| fallback_model.to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String(self.content),
                    extra: Default::default(),
                },
                finish_reason: Some(self.finish_reason.unwrap_or_else(|| "stop".into())),
                logprobs: None,
            }],
            usage: self.usage.unwrap_or_default(),
            extra: Default::default(),
        })
    }
}

/// Index one past the end of the first complete SSE event in `buf`, or `None`
/// for a partial event.
///
/// Per WHATWG HTML §9.2.5 a line ends with CRLF, LF, or CR, and an event ends
/// at a blank line — two consecutive terminators in any mix.
fn find_event_end(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    let mut prev_was_terminator = false;
    while i < buf.len() {
        let term_len = match buf[i] {
            b'\n' => 1,
            b'\r' => {
                if i + 1 >= buf.len() {
                    // Trailing CR after a terminator completes the blank line
                    // whether or not an LF follows in the next packet (push
                    // swallows that LF). Otherwise it only ends a line, and
                    // consuming it here would split one event into two.
                    return if prev_was_terminator {
                        Some(i + 1)
                    } else {
                        None
                    };
                }
                if buf[i + 1] == b'\n' {
                    2
                } else {
                    1
                }
            }
            _ => 0,
        };
        if term_len == 0 {
            prev_was_terminator = false;
            i += 1;
        } else {
            i += term_len;
            if prev_was_terminator {
                return Some(i);
            }
            prev_was_terminator = true;
        }
    }
    None
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Replay a cached response as an OpenAI-style chunk stream: role chunk,
/// content chunk, final chunk with usage, `[DONE]`. `model` echoes the
/// requested name, matching the non-streaming cache path.
pub fn cached_response_to_sse(resp: &ChatResponse, requested_model: &str) -> Vec<Bytes> {
    let id = resp.id.clone();
    let created = resp.created;
    let choice = resp.choices.first();
    let content = choice
        .map(|c| match &c.message.content {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let finish = choice
        .and_then(|c| c.finish_reason.clone())
        .unwrap_or_else(|| "stop".into());

    let mk = |choices: Value, usage: Option<&Usage>| -> Bytes {
        let mut obj = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": choices,
        });
        if let Some(u) = usage {
            obj["usage"] = serde_json::to_value(u).unwrap_or(Value::Null);
        }
        let mut out = b"data: ".to_vec();
        out.extend_from_slice(obj.to_string().as_bytes());
        out.extend_from_slice(b"\n\n");
        Bytes::from(out)
    };

    vec![
        mk(
            json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
            None,
        ),
        mk(
            json!([{"index": 0, "delta": {"content": content}, "finish_reason": null}]),
            None,
        ),
        mk(
            json!([{"index": 0, "delta": {}, "finish_reason": finish}]),
            Some(&resp.usage),
        ),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ]
}

/// OpenAI-style error body, mapping the HTTP status onto OpenAI's `type`
/// vocabulary so SDK client-side error handling behaves.
pub fn openai_error_body(status: u16, msg: &str) -> Value {
    let etype = match status {
        401 | 403 => "authentication_error",
        402 => "insufficient_quota",
        404 => "not_found_error",
        409 => "conflict_error",
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    };
    json!({
        "error": {
            "message": msg,
            "type": etype,
            "param": null,
            "code": null,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(acc: &mut StreamAccumulator, s: &str) {
        acc.push(s.as_bytes());
    }

    #[test]
    fn accumulates_content_and_usage_across_events() {
        let mut acc = StreamAccumulator::new();
        feed(&mut acc, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"qwen2.5:7b\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n");
        feed(&mut acc, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n");
        feed(&mut acc, "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n");
        feed(&mut acc, "data: [DONE]\n\n");

        assert!(acc.saw_done);
        let resp = acc.into_chat_response("fallback").expect("cacheable");
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.created, 123);
        assert_eq!(resp.model, "qwen2.5:7b");
        assert_eq!(
            resp.choices[0].message.content,
            Value::String("Hello".into())
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.total_tokens, 7);
    }

    #[test]
    fn handles_events_split_across_packets_mid_utf8() {
        let mut acc = StreamAccumulator::new();
        // "héllo" split inside the two-byte é sequence.
        let event = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"héllo\"},\"finish_reason\":null}]}\n\n";
        let bytes = event.as_bytes();
        let split = event.find('é').unwrap() + 1; // inside the é bytes
        acc.push(&bytes[..split]);
        assert!(acc.content.is_empty(), "partial event must stay buffered");
        acc.push(&bytes[split..]);
        assert_eq!(acc.content, "héllo");
    }

    #[test]
    fn incomplete_stream_is_not_cacheable() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        );
        assert!(acc.into_chat_response("m").is_none());
    }

    #[test]
    fn empty_content_stream_is_not_cacheable() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        );
        feed(&mut acc, "data: [DONE]\n\n");
        assert!(acc.into_chat_response("m").is_none());
    }

    #[test]
    fn multi_choice_stream_is_not_cacheable() {
        let mut acc = StreamAccumulator::new();
        feed(&mut acc, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}},{\"index\":1,\"delta\":{\"content\":\"b\"}}]}\n\n");
        feed(&mut acc, "data: [DONE]\n\n");
        assert!(acc.into_chat_response("m").is_none());
    }

    #[test]
    fn keepalive_comments_are_ignored() {
        let mut acc = StreamAccumulator::new();
        feed(&mut acc, ": keepalive\n\n");
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n",
        );
        assert_eq!(acc.content, "x");
    }

    #[test]
    fn cached_replay_is_a_valid_openai_chunk_stream() {
        let resp = ChatResponse {
            id: "chatcmpl-99".into(),
            object: "chat.completion".into(),
            created: 42,
            model: "real-model".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String("cached answer".into()),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            },
            extra: Default::default(),
        };
        let chunks = cached_response_to_sse(&resp, "auto");
        assert_eq!(chunks.len(), 4);

        let mut acc = StreamAccumulator::new();
        for c in &chunks {
            acc.push(c);
        }
        assert!(acc.saw_done);
        assert_eq!(acc.model.as_deref(), Some("auto"));
        assert_eq!(acc.content, "cached answer");
        assert_eq!(acc.finish_reason.as_deref(), Some("stop"));
        assert_eq!(acc.usage.as_ref().unwrap().total_tokens, 5);

        for c in &chunks[..3] {
            let s = std::str::from_utf8(c).unwrap();
            let payload = s.strip_prefix("data: ").unwrap().trim();
            let v: Value = serde_json::from_str(payload).unwrap();
            assert_eq!(v["object"], "chat.completion.chunk");
            assert_eq!(v["id"], "chatcmpl-99");
        }
        assert_eq!(&chunks[3][..], b"data: [DONE]\n\n");
    }

    #[test]
    fn unterminated_event_buffer_is_capped() {
        let mut acc = StreamAccumulator::new();
        let junk = format!("data: {}", "A".repeat(2 * 1024 * 1024));
        acc.push(junk.as_bytes());
        assert!(acc.overflowed, "unterminated event must trip the cap");
        assert!(acc.buf.is_empty(), "partial buffer must be released");
        acc.push(b"data: [DONE]\n\n");
        assert!(acc.into_chat_response("m").is_none());
    }

    #[test]
    fn oversized_content_is_not_cached() {
        let mut acc = StreamAccumulator::new();
        let big = "B".repeat(1024 * 1024);
        for _ in 0..9 {
            let ev = format!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{big}\"}}}}]}}\n\n"
            );
            acc.push(ev.as_bytes());
        }
        acc.push(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        );
        acc.push(b"data: [DONE]\n\n");
        assert!(acc.overflowed, "content cap must trip");
        assert!(
            acc.into_chat_response("m").is_none(),
            "an overflowed stream must never be cached"
        );
    }

    #[test]
    fn normal_stream_is_unaffected_by_caps() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
        );
        feed(&mut acc, "data: [DONE]\n\n");
        assert!(!acc.overflowed);
        let r = acc.into_chat_response("m").expect("cacheable");
        assert_eq!(r.choices[0].message.content, Value::String("hello".into()));
    }

    #[test]
    fn crlf_terminated_events_are_parsed() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\r\n\r\n",
        );
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\ndata: [DONE]\r\n\r\n",
        );
        assert_eq!(acc.content, "hi", "CRLF-terminated events must be parsed");
        assert!(acc.saw_done, "CRLF-terminated [DONE] must be recognized");
        assert!(acc.into_chat_response("m").is_some());
    }

    #[test]
    fn cr_only_terminated_events_are_parsed() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\r\rdata: [DONE]\r\r",
        );
        assert_eq!(acc.content, "x");
        assert!(acc.saw_done);
    }

    #[test]
    fn crlf_split_across_packets_terminates_exactly_once() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"y\"},\"finish_reason\":null}]}\r\n\r",
        );
        assert_eq!(acc.content, "y", "blank line CRLF+CR is already complete");
        assert!(!acc.is_mid_event());
        feed(
            &mut acc,
            "\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"z\"},\"finish_reason\":\"stop\"}]}\r\n\r\n",
        );
        assert_eq!(acc.content, "yz", "split-pair LF must not double-terminate");
        assert!(!acc.is_mid_event());
    }

    #[test]
    fn lone_cr_line_ending_split_across_packets_stays_buffered() {
        let mut acc = StreamAccumulator::new();
        feed(&mut acc, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"q\"},\"finish_reason\":\"stop\"}]}\r");
        assert!(
            acc.content.is_empty(),
            "single line terminator is not an event end"
        );
        assert!(acc.is_mid_event());
        feed(&mut acc, "\rdata: [DONE]\r\r");
        assert_eq!(acc.content, "q");
        assert!(acc.saw_done);
    }

    #[test]
    fn oversized_garbage_does_not_grow_buffer_unbounded() {
        let mut acc = StreamAccumulator::new();
        let junk = vec![b'a'; 64 * 1024];
        for _ in 0..64 {
            acc.push(&junk); // 4 MiB total, no newlines at all
        }
        assert!(
            acc.buf.len() <= MAX_EVENT_BUF,
            "accumulator buffer must be capped, got {} bytes",
            acc.buf.len()
        );
        assert!(acc.into_chat_response("m").is_none());
    }

    #[test]
    fn in_band_error_event_makes_stream_uncacheable() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"stop\"}]}\n\n",
        );
        feed(
            &mut acc,
            "data: {\"error\":{\"message\":\"backend blew up\",\"type\":\"server_error\"}}\n\n",
        );
        feed(&mut acc, "data: [DONE]\n\n");
        assert!(
            acc.into_chat_response("m").is_none(),
            "a stream containing an in-band error event must not be cached"
        );
    }

    #[test]
    fn mid_event_boundary_is_tracked() {
        let mut acc = StreamAccumulator::new();
        assert!(!acc.is_mid_event(), "fresh accumulator is at a boundary");
        acc.push(b"data: {\"choices\":[{\"index\":0,\"del");
        assert!(acc.is_mid_event(), "partial event must report mid-event");
        acc.push(b"ta\":{\"content\":\"z\"},\"finish_reason\":null}]}\n\n");
        assert!(!acc.is_mid_event(), "completed event returns to boundary");
        acc.push(b"data: x\r\n");
        assert!(acc.is_mid_event());
        acc.push(b"\r\n");
        assert!(!acc.is_mid_event());
    }

    #[test]
    fn mixed_terminators_do_not_stall_the_parser() {
        let mut acc = StreamAccumulator::new();
        feed(
            &mut acc,
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\rdata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}\r\n\rdata: [DONE]\n\n",
        );
        assert_eq!(
            acc.content, "ab",
            "mixed LF/CR blank lines must terminate events"
        );
        assert!(acc.saw_done);
        assert!(!acc.is_mid_event());
    }

    #[test]
    fn error_body_shape_matches_openai() {
        let v = openai_error_body(400, "bad");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "bad");
        assert!(v["error"].get("param").is_some());
        assert!(v["error"].get("code").is_some());
        assert_eq!(
            openai_error_body(402, "x")["error"]["type"],
            "insufficient_quota"
        );
        assert_eq!(openai_error_body(502, "x")["error"]["type"], "api_error");
    }
}
