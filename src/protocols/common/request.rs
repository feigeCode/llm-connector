//! Common Request Assembly Logic

use crate::error::LlmConnectorError;
use crate::protocols::common::capabilities::EmptyAssistantToolContentStrategy;
use crate::types::{Message, MessageBlock, Role};

fn serialize_empty_assistant_tool_content(
    strategy: EmptyAssistantToolContentStrategy,
) -> serde_json::Value {
    match strategy {
        EmptyAssistantToolContentStrategy::Null => serde_json::Value::Null,
        EmptyAssistantToolContentStrategy::EmptyString => serde_json::json!(""),
        EmptyAssistantToolContentStrategy::EmptyArray => serde_json::json!([]),
    }
}

fn should_use_empty_assistant_tool_content_override(msg: &Message) -> bool {
    msg.role == Role::Assistant && msg.tool_calls.is_some() && msg.content.is_empty()
}

fn partition_openai_visible_blocks(content: &[MessageBlock]) -> (Vec<MessageBlock>, String) {
    let mut thinking_agg = String::new();
    let mut visible = Vec::new();
    for block in content {
        if let MessageBlock::Thinking { thinking, .. } = block {
            if !thinking_agg.is_empty() {
                thinking_agg.push('\n');
            }
            thinking_agg.push_str(thinking);
        } else {
            visible.push(block.clone());
        }
    }
    (visible, thinking_agg)
}

fn openai_export_reasoning_content(msg: &Message, thinking_from_blocks: &str) -> Option<String> {
    let mut out = String::new();
    if let Some(s) = msg.reasoning_content.as_deref() {
        if !s.is_empty() {
            out.push_str(s);
        }
    } else {
        for opt in [
            msg.reasoning.as_deref(),
            msg.thought.as_deref(),
            msg.thinking.as_deref(),
        ] {
            if let Some(s) = opt.filter(|t| !t.is_empty()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(s);
            }
        }
    }
    if !thinking_from_blocks.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(thinking_from_blocks);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Generic message converter for OpenAI-compatible protocols
pub fn openai_message_converter(messages: &[Message]) -> Vec<serde_json::Value> {
    openai_message_converter_with_strategy(
        messages,
        EmptyAssistantToolContentStrategy::EmptyArray,
    )
}

pub fn openai_message_converter_with_strategy(
    messages: &[Message],
    empty_assistant_tool_content_strategy: EmptyAssistantToolContentStrategy,
) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            let (visible, thinking_from_blocks) = partition_openai_visible_blocks(&msg.content);

            let content = if should_use_empty_assistant_tool_content_override(msg) {
                serialize_empty_assistant_tool_content(empty_assistant_tool_content_strategy)
            } else if visible.len() == 1 && visible[0].is_text() {
                serde_json::json!(visible[0].as_text().unwrap())
            } else {
                serde_json::to_value(&visible).unwrap_or(serde_json::json!([]))
            };

            let mut map = serde_json::Map::new();
            map.insert("role".to_string(), serde_json::json!(role));
            map.insert("content".to_string(), content);

            if let Some(ref tc) = msg.tool_calls {
                map.insert("tool_calls".to_string(), serde_json::json!(tc));
            }
            if let Some(ref id) = msg.tool_call_id {
                map.insert("tool_call_id".to_string(), serde_json::json!(id));
            }
            if let Some(ref name) = msg.name {
                map.insert("name".to_string(), serde_json::json!(name));
            }
            if let Some(rc) = openai_export_reasoning_content(msg, &thinking_from_blocks) {
                map.insert("reasoning_content".to_string(), serde_json::json!(rc));
            }

            serde_json::Value::Object(map)
        })
        .collect()
}

/// Downgrade message content for providers that only support text content
pub fn openai_message_converter_downgrade(
    messages: &[Message],
) -> Result<Vec<serde_json::Value>, LlmConnectorError> {
    openai_message_converter_downgrade_with_strategy(
        messages,
        EmptyAssistantToolContentStrategy::EmptyString,
    )
}

pub fn openai_message_converter_downgrade_with_strategy(
    messages: &[Message],
    empty_assistant_tool_content_strategy: EmptyAssistantToolContentStrategy,
) -> Result<Vec<serde_json::Value>, LlmConnectorError> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            let (visible, thinking_from_blocks) = partition_openai_visible_blocks(&msg.content);

            // Downgrade content logic
            let content = if should_use_empty_assistant_tool_content_override(msg) {
                match empty_assistant_tool_content_strategy {
                    EmptyAssistantToolContentStrategy::Null => serde_json::Value::Null,
                    EmptyAssistantToolContentStrategy::EmptyString
                    | EmptyAssistantToolContentStrategy::EmptyArray => serde_json::json!(""),
                }
            } else {
                let content_str = if visible.is_empty() && thinking_from_blocks.is_empty() {
                    String::new()
                } else {
                    let mut text_content = String::new();
                    for block in &visible {
                        if let Some(text) = block.as_text() {
                            text_content.push_str(text);
                        } else {
                            return Err(LlmConnectorError::InvalidRequest(format!(
                                "Provider does not support complex content blocks (found {:?})",
                                block
                            )));
                        }
                    }
                    text_content
                };

                serde_json::json!(content_str)
            };

            let mut map = serde_json::Map::new();
            map.insert("role".to_string(), serde_json::json!(role));
            map.insert("content".to_string(), content);

            if let Some(ref tc) = msg.tool_calls {
                map.insert("tool_calls".to_string(), serde_json::json!(tc));
            }
            if let Some(ref id) = msg.tool_call_id {
                map.insert("tool_call_id".to_string(), serde_json::json!(id));
            }
            if let Some(ref name) = msg.name {
                map.insert("name".to_string(), serde_json::json!(name));
            }
            if let Some(rc) = openai_export_reasoning_content(msg, &thinking_from_blocks) {
                map.insert("reasoning_content".to_string(), serde_json::json!(rc));
            }

            Ok(serde_json::Value::Object(map))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageBlock, Role};

    #[test]
    fn openai_converter_moves_thinking_blocks_to_reasoning_content() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                MessageBlock::thinking_unsigned("internal"),
                MessageBlock::text("hello"),
            ],
            ..Default::default()
        }];
        let out = openai_message_converter_with_strategy(
            &messages,
            EmptyAssistantToolContentStrategy::Null,
        );
        let m = out[0].as_object().unwrap();
        assert_eq!(m["reasoning_content"], "internal");
        assert_eq!(m["content"], "hello");
    }
}
